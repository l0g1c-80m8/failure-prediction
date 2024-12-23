import mujoco
import mujoco.viewer

def main():
    # Load the model from XML file
    model = mujoco.MjModel.from_xml_path('/workspace/code/simulation/model/universal_robots_ur5e/scene.xml')
    
    # Create data instance for simulation
    data = mujoco.MjData(model)
    
    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulate and render
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == '__main__':
    main()