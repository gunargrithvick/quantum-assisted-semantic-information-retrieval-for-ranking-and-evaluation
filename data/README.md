# Local Dataset Directory

Raw datasets are intentionally excluded from version control. Create the
directories below and place the extracted files there:

```text
data/
|-- 20_newsgroups/
|   |-- alt.atheism/
|   `-- ...
`-- reuters21578/
    |-- reut2-000.sgm
    `-- ...
```

See the root [README](../README.md#dataset-setup) for the authoritative
download sources and extraction requirements. For reproducible benchmark
records, save the archive filename, download date, and SHA-256 checksum with
your experiment notes. Do not commit raw datasets or generated model files.
