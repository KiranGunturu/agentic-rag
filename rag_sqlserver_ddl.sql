ALTER DATABASE SCOPED CONFIGURATION
SET PREVIEW_FEATURES = ON;
GO

SELECT
    name,
    value,
    value_for_secondary
FROM sys.database_scoped_configurations
WHERE name = 'PREVIEW_FEATURES';

SELECT @@VERSION;

SELECT
    @@VERSION AS FullVersion,
    SERVERPROPERTY('ProductVersion') AS ProductVersion,
    SERVERPROPERTY('ProductMajorVersion') AS MajorVersion,
    SERVERPROPERTY('ProductLevel') AS ProductLevel,
    SERVERPROPERTY('Edition') AS Edition,
    SERVERPROPERTY('EngineEdition') AS EngineEdition;

CREATE TABLE dbo.hr_policy_docs
(
    id          INT IDENTITY PRIMARY KEY,
    content     NVARCHAR(MAX),
    embedding   VECTOR(1536)
);

CREATE VECTOR INDEX IX_Documents_Embedding
ON dbo.hr_policy_docs(embedding)
WITH
(
    METRIC = 'cosine',
    TYPE = 'DiskANN'
);





