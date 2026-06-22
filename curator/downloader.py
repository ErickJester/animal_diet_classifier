"""
Descargador de imágenes en dos fases.

  Fase A — datasets masivos (Kaggle/HF/Roboflow)
    Baja cada fuente declarada en sources.BULK_SOURCES, la extrae en
    downloads/_raw/<fuente>/ y la reorganiza por dieta en downloads/<dieta>/
    (las especies sin dieta conocida van a downloads/_unsorted/<especie>/).

  Fase B — suplemento con especie verificada (iNaturalist)
    Sólo se habilita DESPUÉS de haber intentado la fase A al menos una vez.
    Para cada especie con dieta conocida que aún no llegue al tope, descarga
    fotos research-grade (especie confirmada por la comunidad).

Anti-repetición (ligera, una sola pasada — sin embeddings):
  · datasets ya hechos se saltan enteros (manifiesto).
  · cada imagen se identifica por su hash MD5; si ya se guardó una idéntica
    (de cualquier fuente), no se vuelve a escribir.
  · las fotos de iNaturalist se deduplican además por id de foto, sin descargar.

La dieta NUNCA se adivina: sale de curator/data/diet_labels.json. La curación
fina (similitud visual) y el reparto train/val son pasos posteriores.
"""

from __future__ import annotations

__version__ = "1.2.0"  # 1.0 fase A+B secuencial · 1.1 ThreadPoolExecutor + reconcile · 1.2 fase C extintos (Commons)

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from . import config
from .manifest import DownloadManifest
from .registry import DietLabels
from .research import research_diet
from .sources import (
    BULK_SOURCES,
    BulkSource,
    download_url,
    fetch_bulk,
    iter_commons_photos,
    iter_inaturalist_photos,
    probe_commons,
    probe_inaturalist,
)

# nombre científico: "Genus" o "Genus species" (con mayúscula inicial)
_SCIENTIFIC_RE = re.compile(r"^[A-Z][a-z]+( [a-z]+)?$")


@dataclass
class PhaseStats:
    saved:    int = 0
    dups:     int = 0
    unsorted: int = 0
    by_diet:  Dict[str, int] = field(default_factory=dict)

    def bump(self, kind: str, diet: Optional[str] = None) -> None:
        setattr(self, kind, getattr(self, kind) + 1)
        if kind == "saved" and diet:
            self.by_diet[diet] = self.by_diet.get(diet, 0) + 1


class DatasetDownloader:
    def __init__(self) -> None:
        self.manifest    = DownloadManifest.load()
        self.diet_labels = DietLabels.load()

    # ── FASE A: datasets masivos ───────────────────────────────────────────────
    def fetch_datasets(self, only: Optional[List[str]] = None) -> None:
        self.manifest.phase_a_attempted = True

        for src in BULK_SOURCES:
            if only and src.name not in only:
                continue

            if not src.enabled and not (only and src.name in only):
                print(f"[{src.name}] deshabilitada (verifica slug/credenciales para activarla)")
                self.manifest.mark_dataset(src.ref, "disabled")
                continue

            if self.manifest.is_dataset_done(src.ref):
                info = self.manifest.datasets[src.ref]
                print(f"[{src.name}] ya descargada ({info.get('images', 0)} imgs) — se salta")
                continue

            print(f"\n[{src.name}] {src.note}")
            print(f"    bajando ({src.kind}: {src.ref}) ...")
            raw_dest = config.RAW_DIR / _safe(src.name)
            path, err = fetch_bulk(src, raw_dest)

            if path is None:
                print(f"    OMITIDA: {err}")
                self.manifest.mark_dataset(src.ref, "skipped")
                self.manifest.save()
                continue

            stats = self._reorganize(path, src)
            self.manifest.mark_dataset(src.ref, "done", images=stats.saved)
            self.manifest.save()
            print(f"    guardadas {stats.saved}  (dup {stats.dups}, sin clasificar {stats.unsorted})")
            if stats.by_diet:
                detalle = "  ".join(f"{d}:{n}" for d, n in sorted(stats.by_diet.items()))
                print(f"    por dieta → {detalle}")

        self.manifest.save()

    def _reorganize(self, raw_path: Path, source: Optional[BulkSource] = None) -> PhaseStats:
        """Recorre las imágenes crudas; la carpeta contenedora es la 'especie'.

        Si la fuente fija `fixed_diet` (p.ej. la clase "other"), toda imagen se
        etiqueta con esa dieta sin pasar por el registro de especies. `max_images`
        (si > 0) corta la fuente al alcanzar ese nº de imágenes guardadas — útil
        para mantener balanceada una clase como "other".
        """
        stats = PhaseStats()
        fixed_diet = source.fixed_diet if source else None
        cap        = source.max_images if source else 0
        for img in sorted(raw_path.rglob("*")):
            if cap and stats.saved >= cap:
                break
            if img.suffix.lower() not in config.IMG_EXTS or not img.is_file():
                continue
            species = img.parent.name
            try:
                data = img.read_bytes()
            except Exception:
                continue
            self._place(data, species=species, stem=img.stem, ext=img.suffix.lower(),
                        stats=stats, known_diet=fixed_diet)
        return stats

    # ── FASE B: suplemento verificado (iNaturalist) ────────────────────────────
    def supplement(self, per_species: Optional[int] = None) -> None:
        if not self.manifest.phase_a_attempted:
            print("La fase 2 (suplemento) se habilita DESPUÉS de la primera descarga de datasets.")
            print("Corre primero:  python download.py datasets")
            return

        target = per_species or config.SUPP_PER_SPECIES
        species_list = self._scientific_species()
        if not species_list:
            print("No hay nombres científicos con dieta en diet_labels.json para suplementar.")
            return

        problema = probe_inaturalist()
        if problema:
            print(f"No puedo usar iNaturalist: {problema}")
            return

        # Reconstruye el estado desde lo que ya está en disco. Hace la descarga
        # reanudable: si una corrida previa se cortó antes de guardar el
        # manifiesto, no se vuelve a bajar ni se duplica lo que ya tenemos.
        recuperadas = self._reconcile_with_disk()
        if recuperadas:
            print(f"(reconciliadas {recuperadas} imágenes ya presentes en disco)")
        self.manifest.save()

        print(f"Suplementando hasta {target} fotos por especie desde iNaturalist (verificadas)\n")
        total = 0
        for species, diet in species_list:
            have = self.manifest.species_count(species)
            if have >= target:
                continue
            need = target - have
            got = self._pull_inaturalist(species, diet, need)
            total += got
            estado = f"+{got}" if got else "sin nuevas"
            print(f"  {species:<28} [{diet:<9}] {estado}  (acumulado {have + got})")
            self.manifest.save()   # incremental: una interrupción pierde, a lo sumo, una especie

        self.manifest.save()
        print(f"\nTotal suplementado en esta pasada: {total}")
        print(f"Imágenes en: {config.DOWNLOADS_DIR}")

    def _reconcile_with_disk(self) -> int:
        """
        Sincroniza el manifiesto con las imágenes ya descargadas en downloads/.

        Recorre las fotos de iNaturalist (nombre 'Genus_species_<photoid>.jpg'),
        registra su id y hash (para no re-bajarlas ni duplicarlas) y reconstruye
        el conteo por especie (para respetar el tope). Idempotente: solo hashea
        las que aún no constaban como vistas.
        """
        counts: Dict[str, int] = {}
        recuperadas = 0
        for diet in config.DIET_CLASSES:
            if diet == "other":
                continue   # 'other' (no-animal) no viene de iNaturalist: sin patrón Genus_species_<id>
            folder = config.DOWNLOADS_DIR / diet
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in config.IMG_EXTS:
                    continue
                parts = f.stem.split("_")
                if not (parts and parts[-1].isdigit()):
                    continue   # no tiene el patrón de foto iNaturalist
                pid, species = parts[-1], " ".join(parts[:-1])
                counts[species] = counts.get(species, 0) + 1
                if not self.manifest.seen_photo("inaturalist", pid):
                    self.manifest.add_photo("inaturalist", pid)
                    try:
                        self.manifest.add_hash(hashlib.md5(f.read_bytes()).hexdigest())
                    except Exception:
                        pass
                    recuperadas += 1
        # el conteo de disco manda (refleja lo realmente colocado)
        for species, n in counts.items():
            if n > self.manifest.species_count(species):
                self.manifest.species[species] = n
        return recuperadas

    def _pull_inaturalist(self, species: str, diet: str, need: int) -> int:
        # 1. Reunir candidatos aún no vistos (paginación rápida, secuencial).
        #    Pedimos de más por si hay duplicados visuales o bajadas fallidas.
        candidates = [
            (pid, url)
            for pid, url in iter_inaturalist_photos(species, max_photos=need * 3)
            if not self.manifest.seen_photo("inaturalist", pid)
        ]
        if not candidates:
            return 0

        # 2. Descargar en paralelo por lotes (la red es el cuello de botella),
        #    pero procesar los resultados en ESTE hilo: así el manifiesto y la
        #    escritura a disco quedan libres de condiciones de carrera.
        got = 0
        stats = PhaseStats()
        workers = config.DOWNLOAD_WORKERS
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(candidates), workers):
                if got >= need:
                    break
                chunk = candidates[start:start + workers]
                for pid, data in pool.map(lambda c: (c[0], download_url(c[1])), chunk):
                    self.manifest.add_photo("inaturalist", pid)  # visto, aun si falló
                    if got >= need or not data:
                        continue
                    if self._place(data, species=species, stem=pid, ext=".jpg",
                                   stats=stats, known_diet=diet) == "saved":
                        self.manifest.add_species(species)
                        got += 1
        return got

    # ── FASE C: animales extintos (Wikimedia Commons + investigación de dieta) ──
    def supplement_extinct(self, per_species: Optional[int] = None) -> None:
        """
        Recopila imágenes de las especies extintas listadas en extinct_species.json.

        Para cada especie:
          · dieta en diet_labels.json  → descarga a downloads/<dieta>/   (origen: known)
          · dieta auto-investigada con confianza alta → downloads/<dieta>/ (suggested)
          · dieta dudosa/desconocida   → downloads/_unsorted/<especie>/  (unknown)

        Las dietas auto-investigadas se vuelcan a diet_suggestions.json para que una
        persona las confirme luego en diet_labels.json (la dieta nunca se da por
        definitiva sin revisión). La descarga es reanudable: la dedup por hash MD5
        evita reescribir contenido ya presente aunque se pierda el manifiesto.
        """
        target = per_species or config.COMMONS_PER_SPECIES
        species_list = self._load_extinct_species()
        if not species_list:
            print(f"No hay especies que recopilar en {config.EXTINCT_SPECIES_PATH}.")
            print("Crea esa lista (un nombre científico por entrada) — ver README, fase C.")
            return

        problema = probe_commons()
        if problema:
            print(f"No puedo usar Wikimedia Commons: {problema}")
            return

        suggestions = self._load_suggestions()
        print(f"Recopilando hasta {target} fotos por especie extinta desde Wikimedia Commons\n")
        total = 0
        for species in species_list:
            diet, origin, guess = self._resolve_extinct_diet(species)
            if guess is not None:   # hubo investigación (suggested o unknown)
                suggestions[species] = {
                    "suggested_diet": guess.diet,
                    "confidence":     round(guess.confidence, 2),
                    "evidence":       guess.evidence,
                    "source":         guess.source_url,
                    "applied":        diet,   # carpeta usada, o null si fue a _unsorted
                }

            have = self.manifest.species_count(species) if diet else 0
            if diet and have >= target:
                print(f"  {species:<26} [{diet:<11}] completa ({have})")
                continue
            need = (target - have) if diet else target
            got  = self._pull_commons(species, diet, need)
            total += got

            etiqueta = diet if diet else "→ _unsorted"
            marca = {"known": "", "suggested": " (dieta sugerida)", "unknown": " (sin dieta)"}[origin]
            print(f"  {species:<26} [{etiqueta:<11}] +{got}{marca}")
            self.manifest.save()   # incremental: una interrupción pierde, a lo sumo, una especie

        self._save_suggestions(suggestions)
        self.manifest.save()
        print(f"\nTotal recopilado en esta pasada: {total}")
        print(f"Imágenes en: {config.DOWNLOADS_DIR}")
        if suggestions:
            print("Dietas auto-investigadas (revísalas y confírmalas en diet_labels.json):")
            print(f"  {config.DIET_SUGGESTIONS_PATH}")

    def _load_extinct_species(self) -> List[str]:
        """Nombres a recopilar desde extinct_species.json. Admite lista u objeto."""
        path = config.EXTINCT_SPECIES_PATH
        if not path.is_file():
            return []
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return []
        if isinstance(data, dict):           # {"species": [...]}  o  {nombre: dieta}
            names = data.get("species") or list(data.keys())
        else:
            names = data
        seen, out = set(), []
        for n in names:
            name = str(n).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
        return out

    def _resolve_extinct_diet(self, species: str):
        """(dieta_o_None, origen, DietGuess_o_None). origen ∈ known|suggested|unknown."""
        known = self.diet_labels.get(species)
        if known in config.DIET_CLASSES:
            return known, "known", None
        guess = research_diet(species)
        if guess.diet and guess.confidence >= config.COMMONS_DIET_MIN_CONF:
            return guess.diet, "suggested", guess
        return None, "unknown", guess

    def _pull_commons(self, species: str, diet: Optional[str], need: int) -> int:
        # Mismo patrón que iNaturalist: reúne candidatos no vistos, baja en paralelo
        # y procesa en este hilo (manifiesto y disco sin condiciones de carrera).
        candidates = [
            (pid, url)
            for pid, url in iter_commons_photos(species, max_photos=need * 3)
            if not self.manifest.seen_photo("commons", pid)
        ]
        if not candidates:
            return 0

        got = 0
        stats = PhaseStats()
        workers = config.DOWNLOAD_WORKERS
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(candidates), workers):
                if got >= need:
                    break
                chunk = candidates[start:start + workers]
                for pid, data in pool.map(lambda c: (c[0], download_url(c[1])), chunk):
                    self.manifest.add_photo("commons", pid)   # visto, aun si falló
                    if got >= need or not data:
                        continue
                    res = self._place(data, species=species, stem=pid, ext=".jpg",
                                      stats=stats, known_diet=diet)
                    if res in ("saved", "unsorted"):
                        if diet:
                            self.manifest.add_species(species)
                        got += 1
        return got

    def _load_suggestions(self) -> Dict[str, dict]:
        path = config.DIET_SUGGESTIONS_PATH
        if not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_suggestions(self, suggestions: Dict[str, dict]) -> None:
        if not suggestions:
            return
        path = config.DIET_SUGGESTIONS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(suggestions, fh, ensure_ascii=False, indent=2, sort_keys=True)

    # ── colocación de una imagen con dedup por hash ────────────────────────────
    def _place(self, data: bytes, species: str, stem: str, ext: str,
               stats: PhaseStats, known_diet: Optional[str] = None) -> str:
        digest = hashlib.md5(data).hexdigest()
        if self.manifest.seen_hash(digest):
            stats.bump("dups")
            return "dup"

        diet = known_diet or self.diet_labels.get(species, species)
        if diet in config.DIET_CLASSES:
            dest_dir = config.DOWNLOADS_DIR / diet
            kind = "saved"
        else:
            dest_dir = config.UNSORTED_DIR / _safe(species)
            diet = None
            kind = "unsorted"

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique(dest_dir, f"{_safe(species)}_{stem}{ext}")
        try:
            dest.write_bytes(data)
        except Exception:
            return "error"

        self.manifest.add_hash(digest)
        stats.bump("saved" if kind == "saved" else "unsorted", diet)
        return "saved" if kind == "saved" else "unsorted"

    # ── utilidades ─────────────────────────────────────────────────────────────
    def _scientific_species(self) -> List[Tuple[str, str]]:
        """Pares (nombre_científico, dieta) válidos para consultar iNaturalist."""
        out = []
        for name, diet in self.diet_labels.raw.items():
            if diet in config.DIET_CLASSES and _SCIENTIFIC_RE.match(name):
                out.append((name, diet))
        return out


def _safe(name: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return cleaned or "desconocida"


def _unique(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    stem, ext = candidate.stem, candidate.suffix
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{i}{ext}"
        i += 1
    return candidate
