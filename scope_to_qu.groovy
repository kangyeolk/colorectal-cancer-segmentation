
import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer

//Aperio Image Scope displays images in a different orientation
//ATTENTION sometimes rotated variable should be True
def rotated = false

def server = QP.getCurrentImageData().getServer()
def h = server.getHeight()
def w = server.getWidth()

// need to add annotations to hierarchy so qupath sees them
def hierarchy = QP.getCurrentHierarchy()

//Prompt user for exported aperio image scope annotation file
//NOTE: CHANGE PATH
String xml_file = System.env['HOME'] + "/Desktop/data/cancer_data/original/P2_0039.xml"
//def file = getQuPath().getDialogHelper().promptForFile(xml_file, null, 'xml_file', null)
//def text = file.getText()
def text = new File(xml_file).getText()
def list = new XmlSlurper().parseText(text)


list.Annotation.each {

    it.Regions.Region.each { region ->

        def tmp_points_list = []

        region.Vertices.Vertex.each{ vertex ->

            if (rotated) {
                X = vertex.@Y.toDouble()
                Y = h - vertex.@X.toDouble()
            }
            else {
                X = vertex.@X.toDouble()
                Y = vertex.@Y.toDouble()
            }
            tmp_points_list.add(new Point2(X, Y))
        }

        def roi = new PolygonROI(tmp_points_list)

        def annotation = new PathAnnotationObject(roi)

        hierarchy.addPathObject(annotation, false)
    }
}