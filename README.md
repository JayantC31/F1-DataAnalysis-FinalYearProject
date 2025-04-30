# *Optimizing car performance in Formula 1 through AI-Driven Telemetry Data Analysis*

A Streamlit app for analyzing Formula 1 telemetry, weather, and tire data using AI/ML techniques.

Jayant Chawla


Please visit website: https://jayant-f1-data-analysis.streamlit.app/

This will show everything

To run code locally: streamlit run streamlit_app.py

Features:

Data Collection: GetData.py and GetExternalData.py are used to fetch and cache session, telemetry, weather, and tire data.

Preprocessing: Align and extract telemetry features, aggregate across laps/races.

Machine Learning techqniues used: PCA, clustering, regression, classification, anomaly detection.

Visualization: Matplotlib-powered charts embedded in a Streamlit UI.

If you only want to run file to collect telemetry or external data from FastF1 API, run using: python GetData.py/GetExternalData.py

Running Tests:

From the project root, run all unit tests: python -m unittest discover tests
Or run a specific test file: python -m unittest tests/test_drivercomparisons.py
