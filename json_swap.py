import os

folder = "2_covariance_parameter_estimation/intermediate"

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("tau_", "gamma_")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

print("Replacement complete.")