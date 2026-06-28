Graph Analysis for Chemical-Protein Interaction Discovery
Project Overview
This repository contains the original code developed by Emanuele Ciuffa as part of his thesis project at the University of L'Aquila (UNIVAQ), titled "Graph Analysis for Chemical-Protein Interaction Discovery".

The primary objective of this software is to analyze biological interactions by leveraging data from the Comparative Toxicogenomics Database (CTD) (available at https://ctdbase.org/). The framework applies graph-based analysis techniques to discover and evaluate complex relationships between chemicals and proteins.

Authorship & Affiliation
- Author: Emanuele Ciuffa
- Institution: University of L'Aquila (Università degli Studi dell'Aquila - UNIVAQ)
- Purpose: Thesis project for the fulfillment of academic requirements.

Intellectual Property & Novelty Notice
This codebase represents entirely original, novel, and unpublished work developed solely for the academic thesis mentioned above.
- Originality: The methodology, specific graph analysis implementations, and integration logic with the CTD repository are the unique intellectual property of the author.
- Usage Restriction: All rights are reserved. No part of this source code may be replicated, redistributed, or used in separate projects without explicit written permission from the author and proper academic citation.

Description
This project consists of a bioinformatics pipeline to study chemicals-proteins interactions by leveraging on graph.
The project is based on Spark 3.5 and Tensorflow: you can run the pipeline on local machine, on a server, on a cluster (for best performance on Hadoop cluster) and link prediction algorithm can run on CPU or GPU (the choice it is automatic and it depends on the availability of GPU).
The database is Neo4j 5.

To start the pipeline you have to run the main script: bio-graph_discovery/src/pipeline.py

Run the main script (pipeline.py) to start the pipeline composed by 4 principal steps:
- Data preparation: extract data from input csv generating a report and the 2 csv files ready to be imported to Neo4j;
- Import graph: create a graph and import data to Neo4j (based on data preparation output);
- Graph analysis: perform graph analysis on Community Detection, Degree Centrality, Connected Components, Shortest Path. Outputs will be generated for each analysis and also a report will be produced;
- Link prediction: perform link prediciton on the graph.

The pipeline is modular and it can be configured to enable the different steps and specific details by config file (in the path graph_discovery/config):
- config-common.yml: enable the step that will be exectued and general configurations;
- config-data_prep.yml: data preparation configurations;
- config-neo4j.yml: neo4j configurations (such as server IP and credential to authenticate);
- config-graph_analysis.yml: enable the analysis you want to perform and specify the vertex (node) as target for Shortest Path (in case it is enabled);
- config-link_prediction.yml: configurations related to the link prediction step.

Other config files:
- config-spark.yml: configure spark parameters;
- config-actions.yml: WARNING: this file should not be altered, because it contains the interactions that have to be parsed form input csv file during the data preparation step.

Input and output path should be configured in the related configuration (yml) file for each step.




*******TECHNOLOGIES*************************************************************************************
Libraries are referenced to the requirements.txt
Main references:
- pyspark 3.5
- tensorflow 2.10.1
- spektral 1.3.1
- numpy 1.23.5
- pyyaml 6.0.3
- dash 4.1.0 / dash-cytoscape  1.0.2 (both with 'pip install dash dash-cytoscape neo4j pandas', then 'pip install dash dash-bootstrap-components')
- matplotlib 3.10.9
- neo4j Desktop 5.26.4 + Neo4j driver 5.28.3 + plugins (apoc-5.26.2-extended.jar and apoc-5.26.24-core.jar)
- cuda 11.8 (conda install --strict-channel-priority -c conda-forge/label/cf202301 cudatoolkit=11.8.0 cudnn=8.8.0.121)

NOTES FOR APOC PLUGIN------------------------
Copy jars to plugins sub-directory of Neo4j dir
Ex: C:\neo4j\neo4j-community-5.26.24\plugins

update file neo4j/conf/neo4j.conf adding:
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
apoc.import.file.enabled=true
apoc.export.file.enabled=true

Restart NEO4J
---------------------------------------------
Additional notes

To enable matplotlib in PyCharm:

- Go to File > Settings (or PyCharm > Preferences).
- In the left-hand menu, expand: Build, Execution, Deployment > Python Debugger.
- Look for the PyQt compatible (or PyQt compatible debugging) option.
- Uncheck it (disable it).
- Click Apply, then OK.
- Go back to File > Settings.
- In the left-hand menu, go to Tools > Python Scientific.
- Uncheck "Show plots in tool window".
- Click Apply, then OK.
- Restart the IDE

*********************************************************************************************************