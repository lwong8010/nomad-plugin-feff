from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import os
import re

import numpy as np
import yaml
from nomad.config import config
from nomad.parsing.parser import MatchingParser

configuration = config.get_plugin_entry_point('feff.parsers:parser_entry_point')

# Column layout constants for FEFF output files
XMU_MIN_COLS = 4  # xmu.dat needs at least: omega k mu mu0
CHI_MIN_COLS = 2  # chi.dat needs at least: k chi
CHI_K2_COL_COUNT = 5  # chi.dat has k²chi in col 4 when this many cols present
ATOMS_MIN_FIELDS = 4  # feff.inp ATOMS line: x y z ipot [tag]
ATOMS_TAG_FIELDS = 5  # need 5 fields to have the element tag
MIN_2D = 2  # ndim check for numpy array

# ---------------------------------------------------------------------------
# Low-level file readers
# ---------------------------------------------------------------------------


def read_xmu_dat(filepath: str) -> dict | None:
    """
    Parse xmu.dat. FEFF xmu.dat columns are:
    omega  k  mu  mu0  chi  |chi|  phase
    We care about: energy (col 0), xmu (col 2), xmu0 (col 3).
    Lines beginning with # are comments/headers.
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim < MIN_2D or data.shape[1] < XMU_MIN_COLS:
            return None
        return {
            'energy': data[:, 0],
            'xmu': data[:, 2],
            'xmu0': data[:, 3],
        }
    except Exception:
        return None


def read_chi_dat(filepath: str) -> dict | None:
    """
    Parse chi.dat. FEFF chi.dat columns are:
    k  chi  |chi|  phase  (and sometimes k²chi in col 4)
    Lines beginning with # are comments/headers.
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim < MIN_2D or data.shape[1] < CHI_MIN_COLS:
            return None
        result = {
            'k': data[:, 0],
            'chi': data[:, 1],
        }
        if data.shape[1] >= CHI_K2_COL_COUNT:
            result['chi_k2'] = data[:, 4]
        else:
            result['chi_k2'] = data[:, 0] ** 2 * data[:, 1]
        return result
    except Exception:
        return None


def get_species_from_feff_inp(feff_inp_path: str) -> str | None:
    """
    Extract the absorbing atom species from feff.inp.
    The absorbing atom is the FIRST entry in the ATOMS section
    with ipot=0.
    Falls back to reading the HOLE card if ATOMS parsing fails.
    Note: feff.inp only contains the local cluster, not the full
    nanoparticle structure — use structure.xyz for that.
    """
    try:
        with open(feff_inp_path) as f:
            lines = f.readlines()

        in_atoms = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('*'):
                continue
            if stripped.upper().startswith('ATOMS'):
                in_atoms = True
                continue
            if in_atoms:
                if stripped.upper().startswith('END'):
                    break
                parts = stripped.split()
                # ATOMS line format: x y z ipot [tag]
                if len(parts) >= ATOMS_MIN_FIELDS:
                    try:
                        ipot = int(parts[3])
                        if ipot == 0 and len(parts) >= ATOMS_TAG_FIELDS:
                            # tag is the element symbol
                            return parts[4]
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def get_species_from_xyz(xyz_path: str, atom_index: int) -> str | None:
    """
    Read element symbol for a given atom index from an XYZ file.
    XYZ format: line 0 = atom count, line 1 = comment, lines 2+ = data.
    Each data line: element x y z
    """
    try:
        with open(xyz_path) as f:
            lines = f.readlines()
        data_lines = lines[2:]
        if atom_index < len(data_lines):
            parts = data_lines[atom_index].strip().split()
            if parts:
                return parts[0]
    except Exception:
        pass
    return None


def atom_index_from_dirname(dirname: str) -> int | None:
    """Extract integer index from directory names like 'atom_0', 'atom_23'."""
    match = re.match(r'atom_(\d+)$', os.path.basename(dirname))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------


def parse_feff_calculation(
    atom_dir: str,
    calc_type: str,
    atom_index: int,
    species: str | None,
    logger: 'BoundLogger',
) -> 'FEFFCalculation':
    """
    Parse one atom_N directory into a FEFFCalculation section.
    calc_type should be 'XANES' or 'EXAFS'.
    """
    from feff.schema_packages.schema_package import CHIData, FEFFCalculation, XMUData

    calc = FEFFCalculation(
        atom_index=atom_index,
        calc_type=calc_type,
    )

    # Resolve species: prefer xyz structure file (set by caller),
    # fall back to feff.inp inside this directory
    if species:
        calc.species = species
    else:
        feff_inp = os.path.join(atom_dir, 'feff.inp')
        if os.path.isfile(feff_inp):
            calc.species = get_species_from_feff_inp(feff_inp)

    if calc.species is None:
        logger.warning('Could not determine species', atom_dir=atom_dir)

    # Parse xmu.dat for XANES
    xmu_path = os.path.join(atom_dir, 'xmu.dat')
    if calc_type == 'XANES' and os.path.isfile(xmu_path):
        xmu = read_xmu_dat(xmu_path)
        if xmu:
            calc.xmu_data = XMUData(
                energy=xmu['energy'],
                xmu=xmu['xmu'],
                xmu0=xmu['xmu0'],
            )
        else:
            logger.warning('Failed to parse xmu.dat', path=xmu_path)

    # Parse chi.dat for EXAFS
    chi_path = os.path.join(atom_dir, 'chi.dat')
    if calc_type == 'EXAFS' and os.path.isfile(chi_path):
        chi = read_chi_dat(chi_path)
        if chi:
            calc.chi_data = CHIData(
                k=chi['k'],
                chi=chi['chi'],
                chi_k2=chi['chi_k2'],
            )
        else:
            logger.warning('Failed to parse chi.dat', path=chi_path)

    return calc


def parse_calc_directory(
    calc_dir: str,
    calc_type: str,
    structure_xyz: str | None,
    logger: 'BoundLogger',
) -> list:
    """
    Walk a XANES/ or EXAFS/ directory and parse all atom_N subdirectories.
    Returns a list of FEFFCalculation objects.
    """
    calculations = []

    if not os.path.isdir(calc_dir):
        logger.warning('Calculation directory not found', path=calc_dir)
        return calculations

    atom_dirs = sorted(
        d
        for d in os.listdir(calc_dir)
        if os.path.isdir(os.path.join(calc_dir, d)) and re.match(r'atom_\d+$', d)
    )

    for dirname in atom_dirs:
        atom_dir = os.path.join(calc_dir, dirname)
        atom_index = atom_index_from_dirname(dirname)

        if atom_index is None:
            logger.warning('Could not parse atom index', dirname=dirname)
            continue

        # Prefer species from the full structure file
        species = None
        if structure_xyz and os.path.isfile(structure_xyz):
            species = get_species_from_xyz(structure_xyz, atom_index)

        calc = parse_feff_calculation(
            atom_dir=atom_dir,
            calc_type=calc_type,
            atom_index=atom_index,
            species=species,
            logger=logger,
        )
        calculations.append(calc)
        logger.info(
            'Parsed calculation',
            calc_type=calc_type,
            atom_index=atom_index,
            species=calc.species,
        )

    return calculations


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------


class FEFFParser(MatchingParser):
    """
    NOMAD parser plugin for FEFF output files.

    Mode 1 (single calculation):
        Triggered when mainfile is xmu.dat or chi.dat directly.
        Produces one FEFFCalculation entry.

    Mode 2 (nanoparticle aggregate):
        Triggered when mainfile is nanoparticle.yaml.
        Produces one FEFFNanoparticleEntry containing all site
        calculations as sub-sections, plus per-element averaged spectra.
    """

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:

        filename = os.path.basename(mainfile)

        if filename == 'nanoparticle.yaml':
            self._parse_nanoparticle(mainfile, archive, logger)
        elif filename == 'xmu.dat':
            self._parse_single_xmu(mainfile, archive, logger)
        elif filename == 'chi.dat':
            self._parse_single_chi(mainfile, archive, logger)
        else:
            logger.warning('Unrecognised mainfile for FEFFParser', mainfile=mainfile)

    # ------------------------------------------------------------------
    # Mode 2 — nanoparticle.yaml
    # ------------------------------------------------------------------

    def _parse_nanoparticle(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        from feff.schema_packages.schema_package import FEFFNanoparticleEntry

        entry_dir = os.path.dirname(mainfile)

        with open(mainfile) as f:
            config_data = yaml.safe_load(f)

        entry = FEFFNanoparticleEntry()
        
        entry._entry_dir = entry_dir

        # --- metadata ---
        meta = config_data.get('metadata', {})
        if 'num_atoms' in meta:
            entry.num_atoms = int(meta['num_atoms'])
        if 'configuration' in meta:
            entry.configuration = int(meta['configuration'])
        if 'morphology' in meta:
            entry.morphology = meta['morphology']
        if 'shell_thickness' in meta:
            entry.shell_thickness = int(meta['shell_thickness'])
        if 'pt_co_ratio' in meta:
            entry.pt_co_ratio = str(meta['pt_co_ratio'])

        # --- structure file ---
        structure_cfg = config_data.get('structure', {})
        structure_file_rel = structure_cfg.get('file')
        structure_xyz = None
        if structure_file_rel:
            structure_xyz = os.path.join(entry_dir, structure_file_rel)
            entry.structure_file = structure_file_rel
            if not os.path.isfile(structure_xyz):
                logger.warning('Structure file not found', path=structure_xyz)
                structure_xyz = None

        # --- FEFF calculations ---
        feff_cfg = config_data.get('feff_calculations', {})

        xanes_cfg = feff_cfg.get('xanes', {})
        if xanes_cfg.get('enabled', True):
            xanes_dir = os.path.join(entry_dir, xanes_cfg.get('path', 'XANES'))
            entry.xanes_calculations = parse_calc_directory(
                xanes_dir, 'XANES', structure_xyz, logger
            )
            logger.info(
                'XANES calculations parsed',
                count=len(entry.xanes_calculations),
            )

        exafs_cfg = feff_cfg.get('exafs', {})
        if exafs_cfg.get('enabled', True):
            exafs_dir = os.path.join(entry_dir, exafs_cfg.get('path', 'EXAFS'))
            entry.exafs_calculations = parse_calc_directory(
                exafs_dir, 'EXAFS', structure_xyz, logger
            )
            logger.info(
                'EXAFS calculations parsed',
                count=len(entry.exafs_calculations),
            )

        archive.data = entry

        # normalize() will run automatically after parse() and will
        # compute the per-element ElementSummary sub-sections
        entry.normalize(archive, logger)

    # ------------------------------------------------------------------
    # Mode 1 — single xmu.dat
    # ------------------------------------------------------------------

    def _parse_single_xmu(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        from feff.schema_packages.schema_package import FEFFCalculation, XMUData

        atom_dir = os.path.dirname(mainfile)
        atom_index = atom_index_from_dirname(atom_dir)
        species = get_species_from_feff_inp(os.path.join(atom_dir, 'feff.inp'))

        xmu = read_xmu_dat(mainfile)
        if xmu is None:
            logger.error('Failed to parse xmu.dat', path=mainfile)
            return

        calc = FEFFCalculation(
            atom_index=atom_index,
            species=species,
            calc_type='XANES',
            xmu_data=XMUData(
                energy=xmu['energy'],
                xmu=xmu['xmu'],
                xmu0=xmu['xmu0'],
            ),
        )
        archive.data = calc

    # ------------------------------------------------------------------
    # Mode 1 — single chi.dat
    # ------------------------------------------------------------------

    def _parse_single_chi(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
    ) -> None:
        from feff.schema_packages.schema_package import CHIData, FEFFCalculation

        atom_dir = os.path.dirname(mainfile)
        atom_index = atom_index_from_dirname(atom_dir)
        species = get_species_from_feff_inp(os.path.join(atom_dir, 'feff.inp'))

        chi = read_chi_dat(mainfile)
        if chi is None:
            logger.error('Failed to parse chi.dat', path=mainfile)
            return

        calc = FEFFCalculation(
            atom_index=atom_index,
            species=species,
            calc_type='EXAFS',
            chi_data=CHIData(
                k=chi['k'],
                chi=chi['chi'],
                chi_k2=chi['chi_k2'],
            ),
        )
        archive.data = calc
