@echo off
:: Define the absolute path to your environment's python executable
SET CONDA_PYTHON="C:\Users\ciuffae\AppData\Local\anaconda3\envs\spark_tres\python.exe"

echo ===================================================
echo Launching Microservices in separate windows...
echo ===================================================

:: 2. Set the PYTHONPATH so your 'src' module imports work properly
SET PYTHONPATH=%~dp0..\..

:: 3. Calculate the script directory (...\src\res_viz)
SET SCRIPT_DIR=%~dp0

:: 4. Run each script forcing the working directory to be the folder the script lives in
START "Main App" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_app.py

START "Degree Centrality" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_degree_centrality.py
START "Degree Centrality View" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_degree_centrality_view.py

START "Connected Components" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_connected_components.py
START "Connected Component Communities View" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_connected_component_communities_view.py

START "Link Prediction" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_link_prediction.py
START "Link Prediction View" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_link_prediction_view.py

START "Shortest Path" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_shortest_path.py
START "Shortest Path View" /D "%SCRIPT_DIR%" cmd /k %CONDA_PYTHON% fe_shortest_path_view.py

echo ===================================================
echo All microservices triggered successfully!
echo ===================================================
pause