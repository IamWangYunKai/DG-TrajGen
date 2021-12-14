import rospy, rostopic


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--input', dest='input_topic', default='None', type=str, help='Input topic.')
    argparser.add_argument('--output', dest='output_fields', default='None', type=str, help='Output fields.')
    argparser.add_argument('--not-repub', action='store_false', help='[Model] Whether to repub.')

    args = argparser.parse_args()
    return args


class RepubField(object):  ## TODO
    def __init__(self, args):
        rospy.init_node('repub_field', anonymous=True)

        input_topic = args.input_topic
        output_fields = args.output_fields

        if not input_topic.startswith('/'):
            input_topic = rospy.get_namespace() + input_topic
        if input_topic.endswith('/'):
            input_topic = input_topic[:-1]

        input_class = None
        rate = rospy.Rate(50)
        while input_class is None and not rospy.is_shutdown():
            input_class, _, _ = rostopic.get_topic_class(input_topic)
            rate.sleep()

        rospy.loginfo('[repub_field] input topic is %s', input_topic)
        if not args.not_repub: topic_sub = rospy.Subscriber(input_topic, input_class, self.callback_topic)
        self.sub = rospy.Subscriber(input_topic, input_class, self.callback)
        self.pubs = []
        self.legal_fields = []

        input_class_instance = input_class()

        output_classes = []
        for field in output_fields:
            if hasattr(input_class_instance, field):
                output_class = type(getattr(input_class_instance, field))
                output_topic = input_topic + '/' + field
                rospy.loginfo('[repub_field] output topic is %s', output_topic)
                self.legal_fields.append(field)
                self.pubs.append( rospy.Publisher(output_topic, output_class, latch=False, queue_size=1) )


    def callback_topic(self, input_msg):
        pass


    def callback(self, input_msg):
        for (field, publisher) in zip(self.legal_fields, self.pubs):
            publisher.publish( getattr(input_msg, field) )


if __name__ == '__main__':
    args = generate_args()
    RepubField(args)
    rospy.spin()
