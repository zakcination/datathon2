import cv2
import os

def match_template(input_image_path, template_path):
    input_image = cv2.imread(input_image_path, 0)
    template = cv2.imread(template_path, 0)

    h_template, w_template = template.shape
    print(f"Template shape: {template.shape}")

    h_img, w_img = input_image.shape
    print(f"Input image shape: {input_image.shape}")

    # Resize template to maintain consistency in size
    scaling_factor = 0.5  # Adjust scaling factor as needed
    template = cv2.resize(template, (int(w_template * scaling_factor), int(h_template * scaling_factor)))
    h_template, w_template = template.shape
    print(f"Resized Template shape: {template.shape}")

    

    methods = [cv2.TM_CCOEFF_NORMED]

    locations = []
    for method in methods:
        img2 = input_image.copy()

        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0] + w_template, location[1] + h_template)
        locations.append((location[0], location[1], w_template, h_template))
        cv2.rectangle(img2, location, bottom_right, 89, 2)  # Consistent border size
        cv2.imshow("Match", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return locations

def draw_matches(input_image_path, locations, output_path):
    image = cv2.imread(input_image_path)
    for (x, y, w, h) in locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Consistent border size
    
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    input_image_path = "data/input_images/apartment_plan.png"
    template_path = "data/templates/hall_table.png"
    output_image_path = "data/output_images/output.png"

    matches = match_template(input_image_path, template_path)
    draw_matches(input_image_path, matches, output_image_path)

    print(f"Output saved at: {output_image_path}")