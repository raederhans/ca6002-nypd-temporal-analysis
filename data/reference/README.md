# Part 2 Reference Geometry

`nypd_police_precincts_simplified.geojson` is a display-optimised derivative of the City of New York Department of City Planning **Police Precincts** dataset, current version 26b.

- Official dataset: https://data.cityofnewyork.us/City-Government/Police-Precincts/y76i-bdw7
- Official data last updated: 26 May 2026
- Geometry: WGS84 longitude/latitude
- Join field: `properties.precinct`
- Transformation: coordinate precision reduction and line simplification for reproducible presentation graphics
- Intended use: visual display only, not high-precision spatial measurement

The simplified file is stored locally so Part 2 can reproduce its map without making a network request or silently changing boundary versions.
