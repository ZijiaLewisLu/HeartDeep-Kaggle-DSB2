import my_utils as mu
import os
import pickle as pk
small_files = ['[N4,T5]<8-12:42:31>.pk']
all_files   = [
 '[N4,T5]<8-12:42:31>.pk', 
 '[N4,T5]<8-12:42:36>.pk', 
 '[N4,T5]<8-12:42:39>.pk', 
 '[N4,T5]<8-12:42:42>.pk', 
 '[N4,T5]<8-12:42:44>.pk', 
 '[N4,T5]<8-12:42:51>.pk', 
][4:]

def get_data( net_type, batch_size,
                 init_states = (), splite_rate=0.1, small=False):
    if small:
        files = small_files
    else:
        files = all_files
    
    files = [ os.path.join('/home/zijia/HeartDeepLearning/DATA/PK/NEW', f) for f in files ]
    
    from RNN.rnn_load import load_rnn_pk
    
    imgs, labels = load_rnn_pk(files)
    
    data_list = mu.prepare_set(imgs, labels, rate=splite_rate)

    if net_type == 'c':
        img_shape = data_list[0].shape[2:]
        data_list = [ d.reshape( (-1,)+img_shape ) for d in data_list]
        
        train, val = mu.create_iter( *data_list, batch_size=batch_size)
        
        return train, val

    elif net_type == 'r':
        
        from rnn.rnn_iter import RIter
        train = RIter( data_list[0], init_states, 
                      label=data_list[1], batch_size=batch_size, last_batch_handle='pad')
        
        val   = RIter( data_list[2], init_states,
                     label=data_list[3],  batch_size=batch_size, last_batch_handle='pad')
        
        return train, val