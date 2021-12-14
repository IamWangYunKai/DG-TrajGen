import rospy, tf

from ..system import nameddict


class ROSPublish(object):
    def __init__(self, pub_dict):
        topics = list(pub_dict.keys())
        self.pub_dict = dict()
        for topic in topics:
            pub = pub_dict[topic]
            publisher = rospy.Publisher(
                name=topic, data_class=pub.data_class,
                latch=pub.latch, queue_size=pub.queue_size)
            self.pub_dict[topic] = (publisher, pub.func)

    def publish(self, topic, msg):
        publisher = self.pub_dict[topic][0]
        func = self.pub_dict[topic][1]
        func(publisher, msg)

PubFormat = nameddict('PubFormat', ('data_class', 'func', 'latch', 'queue_size'))


class ROSSubscribe(object):
    def __init__(self, sub_dict, instance_pointer, *topics):
        ros_namespace = rospy.get_namespace()

        input_topics = []
        for topic in topics:
            if topic.startswith('/'):
                topic = topic[1:]
            input_topics.append(topic)

        self.msg_subscribe_tolerance = rospy.get_param('~msg_subscribe_tolerance', 1.0)
        self.sub_dict = dict()
        self.variable_array = [None]*len(input_topics)
        self.last_valid_time_array = [None]*len(input_topics)

        for topic in input_topics:
            sub = sub_dict[topic]
            subscriber = rospy.Subscriber(
                name=ros_namespace+topic, data_class=sub['data_class'],
                callback=sub['func'], callback_args=(sub, instance_pointer),
                queue_size=sub['queue_size'])
            self.sub_dict[topic] = [subscriber, sub]


    def get(self, topic):
        delta_time = rospy.Time.now().to_sec() - self.sub_dict[topic][1]['last_valid_time']
        # print(self.sub_dict[topic][1]['last_valid_time'], delta_time)
        if delta_time < self.msg_subscribe_tolerance:
            return self.sub_dict[topic][1]['msg_wrap'].reborn(self.sub_dict[topic][1]['variable'])
        else:
            return None

SubFormat = nameddict('SubFormat', ('variable', 'msg_wrap', 'last_valid_time', 'data_class', 'func', 'queue_size'))


def basic_publish(publisher, msg):
    publisher.publish(msg)
