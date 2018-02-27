import csv
import pandas as pd
import scipy.sparse as sp
import sys
import numpy as np

WORD_META_ID = 0
#TOPIC_META_ID = 1
#POS_META_ID = 2
#YEAR_META_ID = 3
#PRODUCT_META_ID = 4

v_words = [ 
            'prime',
            'minister'
            ]

'''v_topics = [
            'AID',
            'BOMB',
            'CRIM',
            'DEF',
            'DIP',
            'DIS',
            'ENT',
            'LAW',
            'LIF',
            'POL',
            'SCI',
            'WEA'
           ]

v_pos = [
            'NN',
            'NNS',
            'NNP',
            'NNPS',
            'RBR',
            'RBS',
            'VB',
            'VBD',
            'VBP',
            'VBZ',
            'VBN',
            'JJ',
            'JJR',
            'JJS'
           ]

v_year = [
            '2003',
            '2004',
            '2005',
            '2006',
            '2007',
            '2008',
            '2009',
            '2010',
            '2011',
            '2012',
            '2013',
            '2014',
            '2015'
           ]


v_product = [
            'G',
            'SPO'
           ]'''



def save_sparse_csr(filename, array):
    np.savez(filename, data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


class Reader():

    def __init__(self,word_meta2_id_file,cooccur_file,metadata_path):

        print("step 1 : load vocabulary and write metadata")
        self.vocab_size, unique_meta, self.meta_vector, word_meta2_id_dict, self.words = self.load_vocabulary_and_write_metadata(word_meta2_id_file,metadata_path)

        self.len_unique_meta = len(unique_meta)

        print("step 2 : load cooccurencies")
        try:
            # try to load serialized coocurrencies
            filename = cooccur_file+"_X.npz"
            self.X = load_sparse_csr(filename)
            self.X_ids = []
            self.X_weights = []    
            for m in range(len(unique_meta)):
                filename = cooccur_file+"_X_ids_"+str(m)+".npz"
                self.X_ids.append(load_sparse_csr(filename))
                filename = cooccur_file+"_X_weights_"+str(m)+".npz"
                self.X_weights.append(load_sparse_csr(filename))
            filename = cooccur_file+"_Y.npy"
            self.Y = np.load(filename)
            print("\n\t ... succesfully loaded serialized coocurrencies")

        except Exception as inst:
            print(str(type(inst)) + "\n" + str(inst.args) + "\n" + str(inst))
            # read coocurrencies from file
            print("\n\t ... read coocurrencies from file")
            self.X, self.X_ids, self.X_weights, self.Y = self.load_cooccurencies(cooccur_file, self.vocab_size, self.meta_vector, unique_meta)

        self.valid_examples_words = self.loadValidSet(word_meta2_id_dict)


    def load_vocabulary_and_write_metadata(self,word_meta2_id_file,metadata_path):
        # LOAD VOCABULARY
        word_meta2_id_df = pd.read_csv(word_meta2_id_file, names=['word','id','meta'], quotechar='"')
        word_meta2_id_df['word'] = word_meta2_id_df['word'].apply(lambda x: str(x).replace('"', ''))
        word_meta2_id_df = word_meta2_id_df.set_index(['word','meta'])
        
        word_meta2_id_df.loc[:,'id'] += 1 # increase the id by one to reserve element 0 for NULL
        word_meta2_id_dict = word_meta2_id_df.T.to_dict('records')[0]
        word_meta2_id_dict[('NULL',-1)] = 0 # add NULL
        vocab_size = len(word_meta2_id_dict)

        # WRITE METADATA        
        metadata_file = open(metadata_path, 'w')
        metadata_file.write('Name\tMeta\n')
        for k,v in sorted(word_meta2_id_dict.items(), key=lambda x: x[1]):
            metadata_file.write(str(k[0])+'\t'+str(k[1])+'\n')
        metadata_file.close()

        words = ['NULL'] + [w[0] for w in word_meta2_id_df.index]
        aux_meta = [w[1] for w in word_meta2_id_df.index]
        unique_meta = sorted(list(set(aux_meta)))
        meta_vector = [-1] + aux_meta

        return vocab_size, unique_meta, meta_vector, word_meta2_id_dict, words

    def load_cooccurencies(self,cooccur_file, vocab_size, meta_vector, unique_meta):

        # LOAD COOCCURRENCIES
        print("\tstep 1 : LOAD COOCCURRENCIES")
        rows = []
        cols = []
        data = []
        y = []
        with open(cooccur_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            row_index = 0
            for row in reader:

                '''if row_index > 100:
                    break'''

                # print(row_index, " ",row)
                current_y = float(row[0])
                y.append(current_y)

                for j in range(1,len(row)):
                    _split = row[j].split(':')
                    col_index = int(_split[0]) + 1 # increase the id by one to reserve element 0 for NULL
                    value = float(_split[1])     
                    rows.append(row_index)
                    cols.append(col_index)
                    data.append(value)
                row_index+=1 
        X = sp.csr_matrix((data, (rows, cols)), shape=(row_index, vocab_size))  
        Y = np.array(y)

        # LOAD FEATURES
        # one for each meta
        print("\tstep 2 : LOAD FEATURES")
        X_weights = []
        X_ids = []

        for m in range(len(unique_meta)):
            rows = []
            cols = []
            sp_weights = []
            sp_ids = []

            max_arbitrary_col_index = 0

            with open(cooccur_file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                row_index = 0
                for row in reader:

                    '''if row_index > 100:
                        break'''

                    # print(row_index, " ",row)
                    current_y = float(row[0])
                    y.append(current_y)
                    
                    arbitrary_col_index = 0
                    for j in range(1,len(row)):
                        _split = row[j].split(':')
                        embedding_index = int(_split[0]) + 1 # increase the id by one to reserve element 0 for NULL
                        value = float(_split[1])   

                        if (meta_vector[embedding_index] == unique_meta[m]):  
                            rows.append(row_index)
                            cols.append(arbitrary_col_index)
                            sp_weights.append(value)
                            sp_ids.append(embedding_index)
                            arbitrary_col_index += 1
                    #print(arbitrary_col_index)

                    if arbitrary_col_index>max_arbitrary_col_index:
                        max_arbitrary_col_index = arbitrary_col_index

                    '''# raw fix for pos and words
                    # ------------------------------------
                    if ( arbitrary_col_index == 1 and (unique_meta[m] == WORD_META_ID or unique_meta[m] == POS_META_ID)):
                        print("ERROR a")
                        sys.exit(-1)
                        rows.append(row_index)
                        cols.append(arbitrary_col_index)
                        old_weight = sp_weights[-1]
                        # print("meta: {}, old_weight : {}".format(m,old_weight))
                        old_weight = 1.
                        sp_weights[-1] = old_weight
                        old_id = sp_ids[-1]
                        sp_weights.append(old_weight)
                        sp_ids.append(old_id)
                        arbitrary_col_index += 1
                    if arbitrary_col_index<max_arbitrary_col_index:
                        print("ERROR b")
                        sys.exit(-1)
                    assert(arbitrary_col_index==2)
                    # ------------------------------------  '''   

                    row_index+=1 

            assert(row_index == len(Y))

            sparse_weight = sp.csr_matrix((sp_weights, (rows, cols)), shape=(row_index, max_arbitrary_col_index))
            sparse_indices = sp.csr_matrix((sp_ids, (rows, cols)), shape=(row_index, max_arbitrary_col_index))
                    
            X_weights.append(sparse_weight)
            X_ids.append(sparse_indices)

        # validity check
        # print(X_ids[1])
        '''for m in range(len(unique_meta)):
            cx = X_ids[m].tocoo()  
            check = {}
            for i,j,v in zip(cx.row, cx.col, cx.data):
                if i in check:
                    check[i]+=1
                else:
                    check[i]=1
            for (k,v) in check.items():
                if (v!=1):
                    print("{} - {}".format(k,v))
                    sys.exit(-1)
        print("\tvalidity check passed")'''

        # step 3 : SERIALIZE MATRICES
        print("\tstep 3 : SERIALIZE MATRICES")
        filename = cooccur_file+"_X"
        save_sparse_csr(filename, X)
        for m in range(len(unique_meta)):
            filename = cooccur_file+"_X_ids_"+str(m)
            save_sparse_csr(filename, X_ids[m])
            filename = cooccur_file+"_X_weights_"+str(m)
            save_sparse_csr(filename, X_weights[m])
        filename = cooccur_file+"_Y"
        np.save(filename, Y)

        return X, X_ids, X_weights, Y


    def loadValidSet(self, word_meta2_id_dict):

        global v_words
        global v_topics
        global v_pos
        global v_year
        global v_product

        valid_set_word = set()
        valid_set_pos = set()

        try :
            valid_examples_words = [word_meta2_id_dict[(w,WORD_META_ID)] for w in v_words]
            valid_set_word = set(valid_examples_words)
        except :
            print(" WARNING: unable to load words ! ")

        '''try :
            valid_examples_topics = [word_meta2_id_dict[(w,TOPIC_META_ID)] for w in v_topics]
            valid_set = valid_set | set(valid_examples_topics)
        except :
            print(" WARNING: unable to load valid TOPICS ! ")'''

        #try :
        #    valid_examples_pos = [word_meta2_id_dict[(w,POS_META_ID)] for w in v_pos]
        #    valid_set_pos = set(valid_examples_pos)
        #except :
        #    print(" WARNING: unable to load valid POS ! ")

        '''try :
            valid_examples_year = [word_meta2_id_dict[(w,YEAR_META_ID)] for w in v_year]
            valid_set = valid_set | set(valid_examples_year)
        except :
            print(" WARNING: unable to load valid YEAR ! ")

        try :
            valid_examples_products = [word_meta2_id_dict[(w,PRODUCT_META_ID)] for w in v_product]
            valid_set = valid_set | set(valid_examples_products)
        except :
            print(" WARNING: unable to load valid PRODUCTS ! ")'''

        valid_examples_words = list(valid_set_word)
        #valid_examples_pos= list(valid_set_pos)
        return valid_examples_words #,valid_examples_pos