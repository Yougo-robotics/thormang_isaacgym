import xml.etree.ElementTree as ET

def add_collision_to_links(urdf_file_path, output_file_path):
    # Parse the URDF XML file
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

    # Iterate through each link element in the URDF
    for link in root.findall("link"):
        link_name = link.get("name")

        # Check if a collision element already exists
        if link.find("collision") is not None:
            continue  # Skip this link if it already has a collision tag

        # Find the visual element to copy its geometry and origin
        visual = link.find("visual")
        if visual is not None:
            # Copy geometry and origin from visual
            geometry = visual.find("geometry")
            origin = visual.find("origin")

            # Create the collision element
            collision = ET.Element("collision")
            collision.set("name", f"collision_{link_name}")

            # Add origin if exists in visual
            if origin is not None:
                collision_origin = ET.SubElement(collision, "origin")
                collision_origin.set("rpy", origin.get("rpy", "0 0 0"))
                collision_origin.set("xyz", origin.get("xyz", "0 0 0"))

            # Add geometry, whether it's a mesh or simpler geometry
            collision_geometry = ET.SubElement(collision, "geometry")
            if geometry.find("mesh") is not None:
                # If geometry is a mesh
                mesh = geometry.find("mesh")
                collision_mesh = ET.SubElement(collision_geometry, "mesh")
                collision_mesh.set("filename", mesh.get("filename"))
                collision_mesh.set("scale", mesh.get("scale", "1 1 1"))
            else:
                # For simpler geometries, e.g., box, cylinder, sphere
                for geom_type in ["box", "cylinder", "sphere"]:
                    simple_geom = geometry.find(geom_type)
                    if simple_geom is not None:
                        collision_simple_geom = ET.SubElement(collision_geometry, geom_type)
                        if geom_type == "box":
                            collision_simple_geom.set("size", simple_geom.get("size"))
                        elif geom_type == "cylinder":
                            collision_simple_geom.set("radius", simple_geom.get("radius"))
                            collision_simple_geom.set("length", simple_geom.get("length"))
                        elif geom_type == "sphere":
                            collision_simple_geom.set("radius", simple_geom.get("radius"))
                        break  # Exit after finding and setting the first geometry type

            # Add any additional elements specific to the collision tag
            sdf_element = ET.SubElement(collision, "sdf")
            sdf_element.set("resolution", "256")

            # Insert the collision element into the link element
            link.append(collision)

    # Write the modified URDF to a new file
    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)
    print(f"Modified URDF saved to {output_file_path}")

# Example usage:
add_collision_to_links("/home/erc/RL_NVIDIA/IsaacGymEnvs/assets/urdf/gogoro/urdf/scooter_V12.urdf", "/home/erc/RL_NVIDIA/IsaacGymEnvs/assets/urdf/gogoro/urdf/scooter_V13.urdf")
