from nomad.config.models.plugins import ParserEntryPoint


class FEFFParserEntryPoint(ParserEntryPoint):
    def load(self):
        from feff.parsers.parser import FEFFParser

        return FEFFParser(**self.model_dump())


parser_entry_point = FEFFParserEntryPoint(
    name='FEFFParser',
    description='Parser for FEFF output files (xmu.dat, chi.dat) and nanoparticle.yaml aggregate entries.',
    mainfile_name_re=r'.*(nanoparticle\.yaml|xmu\.dat|chi\.dat)$',
)
