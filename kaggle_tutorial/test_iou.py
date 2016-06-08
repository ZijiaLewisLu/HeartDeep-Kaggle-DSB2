import iou

FILE_NAME = "o1.pk"

net = iou.load_net()

print "start process..."
iu = iou.process_store(FILE_NAME,net)

