from nomad.config.models.plugins import SchemaPackageEntryPoint


class FEFFSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from feff.schema_packages.schema_package import m_package

        return m_package


schema_package_entry_point = FEFFSchemaPackageEntryPoint(
    name='FEFFSchemaPackage',
    description='Schema package for FEFF parser plugin.',
)
