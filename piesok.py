from build_model import build_graph
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

graph_name = 'znova_11'

skuska = build_graph(graph_name)



skuska.data_shape(tf.float32, (6,), sequences=True)
skuska.embeddings(vocabulary_size=50, embedding_size=2)
skuska.lstm(5)
skuska.relu()
skuska.lstm(2, output='last')
skuska.relu()
skuska.fc(3)
skuska.relu()
skuska.fc(1)
exec(skuska.finish())

step = 0
while step < 1000:
    x = np.random.randint(0,50, size=(5, 6))
    y = np.sum(x, axis=-1).reshape((-1, 1))
    seqlen = 6 * np.ones(5)

    loss, predict, global_step = skuska.step(x, y, sequence_lengths=seqlen)
    step += 1
    if step % 100 == 0:
        print('step', global_step)
        print('loss', loss)
        print('real',y)
        print('predict', predict)
skuska.save()
a = np.array(
    [[1, 2, 3, 4, 5, 6]])
print(skuska.predict(a, sequence_lengths=np.array([2])))

emb = skuska.session.run(skuska.embeddings)
print(emb)
plt.plot(emb[:,0], emb[:,1], 'ro')
plt.show()

skuska.session.close()
