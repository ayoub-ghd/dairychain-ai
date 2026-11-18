import sys
import numpy, pandas, sklearn, plotly, streamlit, yaml

print("Python      :", sys.version.split()[0])
for lib in (numpy, pandas, sklearn, plotly, streamlit, yaml):
    print(f"{lib.__name__:<12}:", lib.__version__)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
print("\nConfig loaded. Data source:", cfg["project"]["data_source"])
print("Lead time mean (assumed):", cfg["procurement"]["lead_time_mean_days"], "days")