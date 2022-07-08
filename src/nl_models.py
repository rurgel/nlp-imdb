from tensorflow.keras import layers, losses, metrics, regularizers
from tensorflow.keras.models import Model

def nn_model(input_dim, output_dim, vocabulary_size):
    if output_dim == 2:
        output_dim = 1

    inputs = layers.Input( shape=(input_dim,))
    x = layers.Embedding( input_dim=vocabulary_size, 
                          output_dim=1024)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D( 1024, 8, 
                       kernel_regularizer=regularizers.L1(0.000001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Flatten()(x)    
    x = layers.Dropout(0.5)(x)
    x = layers.Dense( 64, 
                      kernel_regularizer=regularizers.L1(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense( 32, 
                      kernel_regularizer=regularizers.L1(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)                      
    outputs = layers.Dense( output_dim,
                            kernel_regularizer=regularizers.L1(0.0001),
                            activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile( loss=losses.BinaryCrossentropy(
                            from_logits=False, label_smoothing=0.2), 
                   metrics=metrics.AUC(multi_label=output_dim!=1, name='AUC'),
                   optimizer='adam')
    return model