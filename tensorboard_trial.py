# cd C:\Users\chinm\AppData\Local\conda\conda\envs\tf-gpu
# ./python -m tensorboard.main --logdir D:\MyProjects\PyCharm\tf_kide\Summary

import tensorflow as tf

a = tf.constant(3.0, name="Input")
b = tf.constant(4.0, name="Input")
tf.summary.scalar("Input 1", a)
tf.summary.scalar("Input 2", b)
total = a+b

merged = tf.summary.merge_all()

sess = tf.Session()
summary, total_value = sess.run([merged, total])
print(total_value)

writer = tf.summary.FileWriter('Summary')
writer.add_graph(tf.get_default_graph())
writer.add_summary(summary)
writer.flush()
