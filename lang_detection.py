import numpy as np
import os
import tensorflow as tf
import subprocess
import time
from PIL import Image as pilimg

from loading import DataClass, accuracy, conv_output_size, draw_prediction


# PARAMETRE_NN-------------------------------------------
checkpoints = [70] #[300, 500, 1000]  # if steps == jeden z checkpoints, uloz model a vypis full valid accuracy
batch_size = 1
chunk_size = 4
channels = 3
augm = 1
info_freq = 5
session_log_name = '3_1'  #input('Name your baby... architecture!')
weighted = 1


keep_prob_fc = 0.5

learning_rate = 0.0001

image_height, image_width = (1536, 2048)
output_reshape_size = image_height * image_width

url = os.getcwd()
checkpoint_counter = 0 # pomocna premenna


train_data = DataClass(os.path.join(url, 'train3/'),
                       batch_size, chunk_size,
                       image_height, image_width, augm,
                       data_use='train')
valid_data = DataClass(os.path.join(url, 'valid3/'),
                       batch_size, chunk_size,
                       image_height, image_width, 0,
                       data_use='valid')

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_height, image_width, channels))
    tf_labels = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))

    weights = {
        'conv1': tf.Variable(tf.random_normal(
            [kernel_sizes['conv1'][0], kernel_sizes['conv1'][1], channels, num_filters['conv1']],
            stddev=np.sqrt(2 / (kernel_sizes['conv1'][0] * kernel_sizes['conv1'][1] * channels)))),
        'deconv1': tf.Variable(tf.random_normal(
            [kernel_sizes['conv1'][0], kernel_sizes['conv1'][1], channels, num_filters['conv1']],
            stddev=np.sqrt(2 / (kernel_sizes['conv1'][0] * kernel_sizes['conv1'][1] * channels)))),
        'conv_final': tf.Variable(tf.random_normal(
            [1, 1, 3, 3],
            stddev=np.sqrt(2 / 3)))
    }

    # vytvor vahy pre ostatne konvolucne vrstvy
    for l, l_prev in zip(conv_layer_names[1:], conv_layer_names[:-1]):
        weights[l] = tf.Variable(tf.random_normal(
            [kernel_sizes[l][0], kernel_sizes[l][1], num_filters[l_prev], num_filters[l]],
            stddev=np.sqrt(2 / (kernel_sizes[l][0] * kernel_sizes[l][1] * num_filters[l_prev])))
        )

    # vahy pre cestu spat
    for l, l_prev in zip(conv_layer_names[1:], conv_layer_names[:-1]):
        weights['de'+l] = tf.Variable(tf.random_normal(
            [kernel_sizes[l][0], kernel_sizes[l][1], num_filters[l_prev], num_filters[l]],
            stddev=np.sqrt(2 / (kernel_sizes[l][0] * kernel_sizes[l][1] * num_filters[l_prev])))
        )

    biases = {
        }

    # vytvor biasy pre ostatne konvolucne vrstvy
    for l in conv_layer_names:
        biases[l] = tf.Variable(tf.zeros([num_filters[l]]))
    # Model
    log = []


    def model(data):
        # INPUT je teraz velkosti batch x h x w x ch
        log.append('input: ' + str(data.get_shape().as_list()))
        out = data

        # ak chces menit konvolucne vrstvy, robi sa to hore pod settings
        conv_shapes = [out.get_shape().as_list()]
        # ------convolution-------
        for l in conv_layer_names:
            out = tf.nn.conv2d(out, weights[l], [1, strides[l][0], strides[l][1], 1], padding=paddings[l])
            out = tf.nn.relu(out + biases[l])

            log.append('KERNEL=' + str(kernel_sizes[l]) + ' STRIDE=' + str(strides[l]) + ' ' + paddings[l])
            log.append(l + ': ' + str(out.get_shape().as_list()))
            conv_shapes.append(out.get_shape().as_list())

        # -------------------------

        for i, l in enumerate(reversed(conv_layer_names)):
            out = tf.nn.conv2d_transpose(out, weights['de'+l], conv_shapes[-(i+2)],
                                         [1, strides[l][0], strides[l][1], 1], padding=paddings[l])
            out = tf.nn.relu(out)  # + biases???

            log.append('KERNEL=' + str(kernel_sizes[l]) + ' STRIDE=' + str(strides[l]) + ' ' + paddings[l])
            log.append(l + ': ' + str(out.get_shape().as_list()))

        # -------------------------
        out = tf.nn.conv2d(out, weights['conv_final'], [1, 1, 1, 1], padding="VALID")
        out = tf.nn.relu(out)
        log.append('KERNEL=(1, 1) ' + 'STRIDE=(1, 1) VALID')
        log.append('conv_final' + ': ' + str(out.get_shape().as_list()))

        shape = out.get_shape().as_list()

        print('\n'.join(log))
        return out


    # compute output activations and loss
    output = model(tf_dataset)

    # vahy na loss
    ones_in_output = tf.reduce_sum(tf_labels)
    x = 1 / ones_in_output
    y = 1 / (batch_size* output_reshape_size - ones_in_output)
    loss_weights = 1 + ( batch_size * output_reshape_size * (tf_labels * (x - y) + y) - 1) * weighted

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, tf_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)\
        .minimize(loss)

    prediction = tf.nn.softmax(output)

    s1 = tf.summary.scalar('loss_function_batch', loss)
    s2 = tf.summary.scalar('loss_function_valid', loss)

    # summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    initialization = tf.global_variables_initializer()


with tf.Session(graph=graph) as session:
    step = -1

    # logovanie vysledkov
    file_writer = tf.summary.FileWriter('logs/{}'.format(session_log_name), graph=graph)

    filename_ckpt = "logs/{}/{}.ckpt".format(session_log_name, session_log_name)
    filename_txt = "logs/{}/{}.txt".format(session_log_name, session_log_name)
    if os.path.isfile(filename_txt) and os.stat(filename_txt).st_size == 0:
        os.remove(filename_txt)
    if os.path.isfile(filename_txt):
        try:
            saver.restore(session, filename_ckpt)
        except:
            print("You probably have changed the model architecture."
                  " Please change the 'session_log_name' variable, tooo.")
            session_log_name = input("Type new session_log_name:")
            saver.restore(session, "logs/{}/{}.ckpt".format(session_log_name, session_log_name))
        logfile = open("logs/{}/{}.txt".format(session_log_name, session_log_name), 'r+')
        current_log = logfile.read().split('\n')
        step_0 = int(current_log[0]) + 1
        arch_size = int(current_log[1])
        current_log = current_log[arch_size+1:]
        current_log.reverse()
        logfile.close()
    else:
        session.run(initialization)
        logfile = open("logs/{}/{}.txt".format(session_log_name, session_log_name), 'w')
        logfile.close()
        current_log = []
        step_0 = 1

    print('------------------------')
    print('Training {}'.format(session_log_name))
    print('------------------------')

    (batch_data_valid, batch_labels_valid) = valid_data.next_batch()

    # Timer
    cas = time.time()  # casovac
    subprocess.call(['speech-dispatcher'])  # start speech dispatcher
    continue_training = '1'
    while continue_training == '1':
        step += 1
        # offset = (step * batch_size) % (y_train_ctc.shape[0] - batch_size)
        batch_data, batch_labels = train_data.next_batch()

        feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels}
        _, loss_value, predictions, summary_batch = session.run(
            [optimizer, loss, prediction, s1], feed_dict=feed_dict)

        file_writer.add_summary(summary_batch, global_step=step+step_0)

        if step % info_freq == 0:
            print('STEP {}'.format(step + step_0))
            print('------------------------')
            print('Minibatch loss: {}'.format(loss_value))
            print('------------------------')
            print('Minibatch accuracy:')
            accuracy(predictions, batch_labels)

            valid_predictions, valid_loss, summary_valid = session.run(
                [prediction, loss, s2],
                feed_dict={tf_dataset: batch_data_valid, tf_labels: batch_labels_valid}
            )

            file_writer.add_summary(summary_valid, global_step=step + step_0)
            print('------------------------')
            print('Validation loss: {}'.format(valid_loss))
            print('------------------------')
            print('Validation accuracy:')
            accuracy(valid_predictions, batch_labels_valid)
            print('====================================================================')



        if step == checkpoints[checkpoint_counter]:
            print("{} steps took {} minutes.".format(checkpoints[checkpoint_counter], (time.time()-cas)/60))

            if step != 0:
                current_log_local = []
                current_log_local.append('Minibatch loss at step {}: {}'.format(step + step_0, loss_value))
                current_log_local.append('Minibatch accuracy: ')
                current_log_local += accuracy(predictions, batch_labels, False)
                current_log_local.append('------------------------')
                current_log_local.append('Validation loss at step {}: {}'.format(step + step_0, valid_loss))
                current_log_local.append('Validation accuracy: ')
                current_log_local += accuracy(valid_predictions, batch_labels_valid, False)
                current_log_local.append("Elapsed time: {} minutes". format((time.time()-cas)/60))
                current_log_local.append('                                                        ')
                current_log_local.append('========================================================')
                current_log_local.reverse()
                current_log += (current_log_local)

            # uloz checkpoint
            save_path = saver.save(session, "{}/logs/{}/{}.ckpt".format(url, session_log_name, session_log_name))
            print('Checkpoint saved at total step {}.'.format(
                                     step + step_0))
            # nastav dalsi checkpoint alebo ukonci
            checkpoint_counter +=1
            if checkpoint_counter >= len(checkpoints):
                continue_training = '0'
                subprocess.call(['spd-say',
                                 'Ooo!'.format(
                                     step + step_0)])
            else:
                subprocess.call(['spd-say',
                                 't.'.format(
                                     step + step_0)])



    results = []
    valid_labels = []
    for offset in range(0, valid_data.total_data_size - batch_size + 1, batch_size):
        data, lab = valid_data.next_batch()

        predict = ((session.run(
            prediction,
            feed_dict={tf_dataset: data}
        )))
        results.append(predict)
        valid_labels.append(lab)



results = np.array(results)
valid_labels = np.array(valid_labels)
draw_prediction(results, valid_labels)


current_log.append('------------------------------------------------------')
print("validation results shape: ", results.shape)
print('Validation accuracy (full) after {} steps: '.format(step+step_0))
current_log += accuracy(results, valid_labels)
current_log.append('Validation accuracy (full) after {} steps: '.format(step+step_0))
print('                                                                    ')
print('====================================================================')


log.append('batch size: {}'.format(batch_size))
log.append('learning rate: {}'.format(learning_rate))
current_log.reverse()
logfile = open('logs/{}/{}.txt'.format(session_log_name, session_log_name), 'w')
logfile.write(str(step + step_0)+'\n')
logfile.write(str(len(log)+3)+'\n')
logfile.write('\n'.join(log) + '\n\n\n')
logfile.write('\n'.join(current_log))
logfile.close()

file_writer.close()