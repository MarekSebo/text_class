from build_model import build_graph
import numpy as np
import tensorflow as tf

skuska = build_graph()

skuska.data_shape(tf.float32, (10, 5))
skuska.lstm(3)
skuska.relu()
skuska.fc(1)
skuska.finish()

step = 0
while step < 15000:
    x = np.random.randint(0,10, size=(5, 10, 5))
    y = np.sum(np.sum(x, axis=-1), axis=-1).reshape((-1, 1))

    loss, predict = skuska.step(x, y)
    step += 1
skuska.session.close()