import mujoco as mj
import os

# Set headless rendering mode
os.environ["MUJOCO_GL"] = "osmesa" #"osmesa"  # or "egl"

# Minimal MuJoCo model
xml = """
<mujoco>
    <worldbody/>
</mujoco>
"""

try:
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
    print("MuJoCo context initialized successfully.")
except mj.FatalError as e:
    print(f"MuJoCo FatalError: {e}")