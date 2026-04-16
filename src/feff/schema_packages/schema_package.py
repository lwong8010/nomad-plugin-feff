from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import os
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
        self._populate_results(archive, logger)

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
        
    def _populate_results(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        import os
        import numpy as np
        from nomad.datamodel.results import (
            Results, Material, Method, Simulation, ElementalComposition,
            Structure, Structures, Spectra, SpectroscopicProperties,
            Properties, OptimadeSpecies,
        )

        if archive.results is None:
            archive.results = Results()
        if archive.results.properties is None:
            archive.results.properties = Properties()

        # --- Material: composition ---
        material = Material()
        element_counts: dict[str, int] = {}
        for summary in (self.element_summaries or []):
            if summary.element and summary.site_count:
                element_counts[summary.element] = summary.site_count

        if element_counts:
            total = sum(element_counts.values())
            material.elements = list(element_counts.keys())
            material.elemental_composition = [
                ElementalComposition(
                    element=el,
                    atomic_fraction=count / total,
                )
                for el, count in element_counts.items()
            ]

        # --- Material: atomic structure from XYZ ---
        if self.structure_file:
            structure_path = self._resolve_structure_path(archive)
            if structure_path:
                try:
                    from ase.io import read as ase_read
                    atoms = ase_read(structure_path)
                    n = len(atoms)

                    # Species subsections (one per unique element)
                    unique_species = []
                    for sym in sorted(set(atoms.get_chemical_symbols())):
                        unique_species.append(
                            OptimadeSpecies(name=sym, chemical_symbols=[sym])
                        )

                    structure = Structure(
                        dimension_types=[1, 1, 1],
                        nperiodic_dimensions=3,
                        n_sites=n,
                        lattice_vectors=atoms.get_cell()[:] * 1e-10,   # Å → m
                        cartesian_site_positions=atoms.get_positions() * 1e-10,
                        species_at_sites=atoms.get_chemical_symbols(),
                        species=unique_species,
                    )
                    archive.results.properties.structures = Structures(
                        structure_original=structure
                    )
                    logger.info(
                        'Structure parsed',
                        n_atoms=n,
                        formula=atoms.get_chemical_formula(),
                    )
                except Exception as e:
                    logger.warning('Failed to parse structure file', error=str(e))

        archive.results.material = material

        # --- SpectroscopicProperties: per-element mean spectra ---
        spectra_list = []
        for summary in (self.element_summaries or []):
            # Mean XANES
            if summary.mean_xmu and summary.mean_xmu.energy is not None:
                spectra_list.append(Spectra(
                    type='XANES',
                    label='computation',
                    n_energies=len(summary.mean_xmu.energy),
                    energies=summary.mean_xmu.energy * 1.60218e-19,  # eV → J
                    intensities=summary.mean_xmu.xmu,
                    intensities_units='a.u.',
                ))
            # Mean EXAFS — chi(k) has no energy axis, skip energies field
            if summary.mean_chi and summary.mean_chi.k is not None:
                spectra_list.append(Spectra(
                    type='EXAFS',
                    label='computation',
                    n_energies=len(summary.mean_chi.k),
                    intensities=summary.mean_chi.chi,
                    intensities_units='a.u.',
                ))

        if spectra_list:
            archive.results.properties.spectroscopic = SpectroscopicProperties(
                spectra=spectra_list
            )

        # --- Method ---
        method = Method()
        simulation = Simulation()
        simulation.program_name = 'FEFF'
        calc_types = []
        if self.xanes_calculations:
            calc_types.append('XANES')
        if self.exafs_calculations:
            calc_types.append('EXAFS')
        if calc_types:
            simulation.program_version = ', '.join(calc_types)
        method.simulation = simulation
        archive.results.method = method

        logger.info(
            'Results populated',
            elements=list(element_counts.keys()),
            calc_types=calc_types,
        )

    def _resolve_structure_path(self, archive: 'EntryArchive') -> 'str | None':
        import os
        try:
            entry_dir = os.path.dirname(
                archive.m_context.raw_path(archive.metadata.mainfile)
            )
            path = os.path.join(entry_dir, self.structure_file)
            return path if os.path.isfile(path) else None
        except Exception:
            return None


m_package.__init_metainfo__()
