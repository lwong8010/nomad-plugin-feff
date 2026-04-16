from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np
from nomad.config import config
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import MEnum, Quantity, SchemaPackage, Section, SubSection

configuration = config.get_plugin_entry_point(
    'feff.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


class XMUData(ArchiveSection):
    """Parsed contents of xmu.dat — XANES absorption spectrum."""

    energy = Quantity(
        type=np.float64,
        shape=['*'],
        unit='eV',
        description='Energy grid for the XANES spectrum.',
    )
    xmu = Quantity(
        type=np.float64,
        shape=['*'],
        description='Absorption mu(E) at each energy point.',
    )
    xmu0 = Quantity(
        type=np.float64,
        shape=['*'],
        description='Atomic background mu0(E).',
    )


class CHIData(ArchiveSection):
    """Parsed contents of chi.dat — EXAFS chi(k) spectrum."""

    k = Quantity(
        type=np.float64,
        shape=['*'],
        unit='1/angstrom',
        description='Wavenumber grid.',
    )
    chi = Quantity(
        type=np.float64,
        shape=['*'],
        description='EXAFS chi(k).',
    )
    chi_k2 = Quantity(
        type=np.float64,
        shape=['*'],
        description='k²-weighted chi(k).',
    )


class FEFFCalculation(ArchiveSection):
    """
    Represents a single FEFF calculation for one absorbing site.
    Used as a standalone entry (Mode 1) or as a sub-section (Mode 2).
    """

    m_def = Section()

    atom_index = Quantity(
        type=int,
        description='Index of the absorbing atom (from the atom_N directory name).',
    )
    species = Quantity(
        type=str,
        description='Element symbol of the absorbing atom (e.g. Pt, Co).',
    )
    calc_type = Quantity(
        type=MEnum('XANES', 'EXAFS'),
        description='Type of FEFF calculation.',
    )
    xmu_data = SubSection(section_def=XMUData)
    chi_data = SubSection(section_def=CHIData)


class ElementSummary(ArchiveSection):
    """
    Per-element averaged spectra, computed by normalize() on FEFFNanoparticleEntry.
    """

    element = Quantity(
        type=str,
        description='Element symbol (e.g. Pt, Co).',
    )
    site_count = Quantity(
        type=int,
        description='Number of absorbing sites of this element.',
    )
    site_indices = Quantity(
        type=int,
        shape=['*'],
        description='Atom indices contributing to this average.',
    )
    mean_xmu = SubSection(
        section_def=XMUData,
        description='Mean XANES spectrum across all sites of this element.',
    )
    mean_chi = SubSection(
        section_def=CHIData,
        description='Mean EXAFS spectrum across all sites of this element.',
    )


class FEFFNanoparticleEntry(EntryData):
    """
    Aggregate entry representing a single nanoparticle.
    Contains multiple FEFFCalculation sub-sections (one per absorbing site)
    and per-element averaged spectra computed by normalize().
    Triggered by the presence of a nanoparticle.yaml file.
    """

    m_def = Section()

    # --- metadata from nanoparticle.yaml ---
    num_atoms = Quantity(
        type=int,
        description='Total number of atoms in the nanoparticle.',
    )
    configuration = Quantity(
        type=int,
        description='Configuration index.',
    )
    morphology = Quantity(
        type=MEnum('random', 'core_shell'),
        description='Nanoparticle morphology.',
    )
    shell_thickness = Quantity(
        type=int,
        description='Shell thickness in Angstrom (core-shell only).',
    )
    pt_co_ratio = Quantity(
        type=str,
        description='Pt:Co ratio string, e.g. "1:1".',
    )
    structure_file = Quantity(
        type=str,
        description='Path to the structure file (.xyz) relative to this entry.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
    )

    # --- per-site calculations ---
    xanes_calculations = SubSection(
        section_def=FEFFCalculation,
        repeats=True,
        description='One FEFFCalculation per absorbing site from the XANES directory.',
    )
    exafs_calculations = SubSection(
        section_def=FEFFCalculation,
        repeats=True,
        description='One FEFFCalculation per absorbing site from the EXAFS directory.',
    )

    # --- per-element averages (populated by normalize) ---
    element_summaries = SubSection(
        section_def=ElementSummary,
        repeats=True,
        description='Per-element averaged spectra, computed during normalization.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        self._compute_element_summaries(logger)

    def _compute_element_summaries(self, logger: 'BoundLogger') -> None:
        """Group calculations by element and compute mean spectra."""
        from collections import defaultdict

        groups: dict[str, dict] = defaultdict(
            lambda: {'xanes': [], 'exafs': [], 'indices': []}
        )

        for calc in self.xanes_calculations or []:
            if calc.species and calc.xmu_data:
                groups[calc.species]['xanes'].append(calc.xmu_data)
                groups[calc.species]['indices'].append(calc.atom_index)

        for calc in self.exafs_calculations or []:
            if calc.species and calc.chi_data:
                groups[calc.species]['exafs'].append(calc.chi_data)

        summaries = []
        for element, data in groups.items():
            summary = ElementSummary(
                element=element,
                site_count=len(data['indices']),
                site_indices=data['indices'] if data['indices'] else None,
            )

            if data['xanes']:
                summary.mean_xmu = self._mean_xmu(data['xanes'], logger)

            if data['exafs']:
                summary.mean_chi = self._mean_chi(data['exafs'], logger)

            summaries.append(summary)
            logger.info(
                'ElementSummary computed',
                element=element,
                site_count=summary.site_count,
            )

        self.element_summaries = summaries

    def _mean_xmu(self, xmu_list: list, logger: 'BoundLogger') -> XMUData | None:
        """Interpolate all xmu.dat onto a common energy grid and average."""
        try:
            ref_energy = xmu_list[0].energy
            xmu_arrays = [np.interp(ref_energy, x.energy, x.xmu) for x in xmu_list]
            return XMUData(
                energy=ref_energy,
                xmu=np.mean(xmu_arrays, axis=0),
            )
        except Exception as e:
            logger.warning('Failed to compute mean xmu', error=str(e))
            return None

    def _mean_chi(self, chi_list: list, logger: 'BoundLogger') -> CHIData | None:
        """Interpolate all chi.dat onto a common k grid and average."""
        try:
            ref_k = chi_list[0].k
            chi_arrays = [np.interp(ref_k, c.k, c.chi) for c in chi_list]
            return CHIData(
                k=ref_k,
                chi=np.mean(chi_arrays, axis=0),
            )
        except Exception as e:
            logger.warning('Failed to compute mean chi', error=str(e))
            return None


m_package.__init_metainfo__()
