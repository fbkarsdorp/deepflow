import tempfile

from typing import List

import cairosvg
import lxml.etree


def create_image_file(lines: List[str], image_template: str, y=74, y_step=20) -> str:
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    svg = lxml.etree.parse(image_template).getroot()
    text_element = svg.find(".//svg:text[@id='Type-something']", namespaces=namespace)
    # clean any lines in the image
    for child in text_element.iterchildren():
        text_element.remove(child)

    for line in lines:
        element = lxml.etree.Element('tspan', nsmap=namespace)
        element.attrib['x'] = '55'
        element.attrib['y'] = str(y)
        element.text = line
        text_element.append(element)
        y += y_step

    svg = lxml.etree.tostring(svg, standalone=True, xml_declaration=True)
    image_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cairosvg.svg2png(bytestring=svg, write_to=image_file.name)
    image_file.close()
    return image_file.name
    

if __name__ == '__main__':
    text = ["when sneezing baby it nothing do your thang i ain't gon' stop you girl",
            'all in your mini skirt really squirt, show me something baby',
            "once i give you what really hurts, you won't owe me nothing (come on!)"]
    image_file = create_image_file(text)
    with open(image_file, 'rb') as f:
        print(f.read())
    os.unlink(image_file)
    print(os.path.exists(image_file))
