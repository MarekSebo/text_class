from build_model import build_graph
import numpy as np
import tensorflow as tf


skuska = build_graph('znova_2')

skuska.data_shape(tf.float32, (3, 2), sequences=True)
# skuska.embeddings(vocabulary_size=50, embedding_size=8)
# skuska.out = tf.reshape(skuska.out, shape=(-1, skuska.current_shape[1] * skuska.current_shape[2], skuska.current_shape[3]))
skuska.lstm(5)
skuska.relu()
skuska.lstm(2, output='last')
skuska.relu()
skuska.fc(3)
skuska.relu()
skuska.fc(1)
skuska.finish()

step = 0
while step < 500:
    x = np.random.randint(0,50, size=(5, 3, 2))
    y = np.sum(np.sum(x, axis=-1), axis=-1).reshape((-1, 1))
    seqlen = 3 * np.ones(5)

    loss, predict, global_step = skuska.step(x, y, sequence_lengths=seqlen)
    step += 1
    if step % 100 == 0:
        print('step', global_step)
        print('loss', loss)
        print('real',y)
        print('predict', predict)
skuska.save()
a = np.array(
    [[[1, 2], [3, 4], [5, 6]]])
print(skuska.predict(a, sequence_lengths=np.array([2])))
skuska.session.close()
