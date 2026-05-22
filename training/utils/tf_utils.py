import numpy as np
import tensorflow as tf


def load_model(path_to_frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def run_inference_for_batch(batch, session):
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_scores', 'detection_boxes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: batch})
    output_dict['num_detections'] = output_dict['num_detections'].astype(int)
    img_height, img_width = batch.shape[1:3]
    output_dict['detection_boxes'] = (output_dict['detection_boxes'] * [img_height, img_width,
                                                                        img_height, img_width]).astype(int)
    return output_dict
