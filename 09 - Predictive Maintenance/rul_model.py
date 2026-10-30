import keras
import keras.backend
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling1D,
    concatenate,
)
from keras.models import Model, load_model
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler

from pdm.config import RULConfig, RULDefaults

# nb_features is the number of features in your data
# nb_out is ... always 1 and we don't know why...?


class RULModel:
    """
    A class for Remaining Useful Life predictions using a CNN and Long Short-Term Memory.

    Methods:
    - batchTrain(self, train_data, target_column, train_columns, save_filename): Trains the model using training data.
    - batchPredict(self, test_data, used_columns): Makes predictions using a Pandas Dataframe of data.
    """

    # Model can be trained using a set of time windows
    # ranging from 20 to 120. Specify the window length

    def __init__(
        self,
        lstm_units: int = RULDefaults.LSTM_UNITS,
        conv_filters: int = RULDefaults.CONV_FILTERS,
        kernel_size: int = RULDefaults.KERNEL_SIZE,
        training_epochs: int = RULDefaults.TRAINING_EPOCHS,
        training_patience: int = RULDefaults.TRAINING_PATIENCE,
        validation_split: int = RULDefaults.VAL_SPLIT,
        attention_width: int = RULDefaults.ATTENTION_WIDTH,
        regulizer_l1: float = RULDefaults.REGULIZER_L1,
        regulizer_l2: float = RULDefaults.REGULIZER_L2,
        attention_regulizer_weight: float = RULDefaults.ATT_REG_WEIGHT,
        optimizer_learning_rate: float = RULDefaults.OPTIMIZER_LEARNING_RATE,
        sequence_length: int = RULDefaults.SEQ_LENGTH,
        asset_id_column: str = RULDefaults.ASSET_ID_COL,
        cycle_column: str = RULDefaults.CYCLE_COL,
        sensor_columns: list[str] = [],
        max_RUL: int = RULDefaults.MAX_RUL,
        target_column: str = RULDefaults.TARGET_COLUMN,
        save_file_name: str = RULDefaults.SAVE_FILE,
        training_callbacks: list[str] = [],
        **kwargs,
    ) -> None:
        """
        Constructor for the Attention_Multihead_CNLSTM_RUL_Predictor_Model class.

        Parameters:
        - lstm_units (int): The number of long short-term memory units
        - conv_filters (int): The number of convolutional filters
        - kernel_size (int): The size of the kernel
        - dropout_rate (float):
        """
        self.lstm_units = lstm_units
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.training_epochs = training_epochs
        self.training_patience = training_patience
        self.validation_split = validation_split
        self.attention_width = attention_width
        self.regulizer_l1 = regulizer_l1
        self.regulizer_l2 = regulizer_l2
        self.attention_regulizer_weight = attention_regulizer_weight
        self.optimizer_learning_rate = optimizer_learning_rate
        self.sequence_length = sequence_length
        self.asset_id_column = asset_id_column
        self.cycle_column = cycle_column
        self.sensor_columns = sensor_columns
        self.max_RUL = max_RUL
        self.target_column = target_column
        self.save_file_name = save_file_name
        self.training_callbacks = training_callbacks
        self.nb_features = (
            len(sensor_columns) + 1
        )  # Number of sensors + cycle_norm = number of features

        self.is_trained = False
        self.model = None
        self.cycle_norm = "cycle_norm"

    def batch_train(self, train_data, training_callbacks=[]):
        # Prepare the training data
        train_data = self.__prepare_training_data(train_data)

        # Normalize the data
        norm_train_data = self._data_scaler(train_data)

        # Specify number of timesteps, features, and output from the training data
        if self.cycle_norm not in self.sensor_columns:
            self.sensor_columns.append(self.cycle_norm)
        seq_array = self._sequence_training_gen(
            norm_train_data,
            self.sequence_length,
            self.sensor_columns,
        )

        # self.nb_features = seq_array.shape[2]

        label_array = self._label_generator(
            norm_train_data, self.sequence_length, [self.target_column]
        )

        training_sample_multihead = [
            seq_array[:, :, i].reshape((seq_array.shape[0], self.sequence_length, 1))
            for i in range(self.nb_features)
        ]

        self._build_model()

        # Creat model checkpoint to save the best model performance
        model_checkpoint = ModelCheckpoint(
            filepath=self.save_file_name, save_best_only=True, verbose=1
        )

        # specify early_callback to stop training model if there's no improvement
        early_stopping = EarlyStopping(patience=self.training_patience)

        # Get the model summary
        self.model.summary()

        # Train the multihead model
        #   - The Callback above will save the best model into the file
        training_callbacks = (
            [
                early_stopping,
                model_checkpoint,
            ]
            + training_callbacks
            + self.training_callbacks
        )
        train_history = self.model.fit(
            training_sample_multihead,
            label_array,
            epochs=self.training_epochs,
            validation_split=self.validation_split,
            callbacks=training_callbacks,
        )

        self.is_trained = True

    def load_model_from_disk(self):
        self.__load_model_from_disk(self.save_file_name)
        return self.model

    def batch_predict(self, test_data):
        if self.is_trained is False:
            raise ValueError(
                "The model is not trained.  The save_file_name in the configuration may be incorrect."
            )

        # Prepare the test data
        test_data = self.__prepare_test_data(test_data)

        # Normalize the test data
        features_scaled = self._data_scaler(test_data)

        if self.cycle_norm not in self.sensor_columns:
            self.sensor_columns.append(self.cycle_norm)

        seq_array_test, label_array_test = self._test_data_eval_prep(
            features_scaled, self.sequence_length, self.sensor_columns
        )

        test_sample_multihead = test_data = [
            seq_array_test[:, :, i].reshape(
                (seq_array_test.shape[0], self.sequence_length, 1)
            )
            for i in range(self.nb_features)
        ]

        # Evaluate the model on test data
        # model_evaluation_res = self.model.evaluate(
        #     test_sample_multihead, label_array_test, return_dict=True
        # )

        # Make predictions using your model
        predictions = self.model.predict(test_data)

        # Return predictions as Pandas Series
        return pd.Series(predictions.flatten(), name="predictions")

    def __load_model_from_disk(self, filename):
        """Function to load the best model after training"""
        self.is_trained = True
        self.model = load_model(
            filename,
            custom_objects={
                "_rmse": self._rmse,
                "_k_score": self._k_score,
                "SeqSelfAttention": SeqSelfAttention,
            },
        )

    def __prepare_training_data(self, train_data: pd.DataFrame):
        """
        Function to prepare the training data.  It sorts the dataframe and sets the RUL to the actual or max RUL.

        Args:
            train_data (pd.DataFrame): The Dataframe containing the training data.  It should have an ID and cycle
                column and for training, the data must be complete, meaning the last cycle is when RUL = 0.


        Returns:
            The prepared dataframe

        """
        # Sort the data and group it by ID
        train_data = train_data.sort_values([self.asset_id_column, self.cycle_column])
        rul = pd.DataFrame(
            train_data.groupby(self.asset_id_column)[self.cycle_column].max()
        ).reset_index()

        # Now for training data, the maximum cycle is the life of the component.
        # So RUL = MaxCycle - CurrentCycle
        rul.columns = [self.asset_id_column, "max"]
        train_data = train_data.merge(rul, on=[self.asset_id_column], how="left")

        train_data["RUL"] = train_data["max"] - train_data[self.cycle_column]
        train_data.drop("max", axis=1, inplace=True)

        # Set R_early (the RUL estimate) to be the smaller of:
        #       A) the actual RUL
        #       B) the max RUL we assigned above
        train_data[self.target_column] = train_data["RUL"].apply(
            lambda x: self.max_RUL if x >= self.max_RUL else x
        )
        train_data.drop("RUL", axis=1, inplace=True)
        return train_data

    def __prepare_test_data(self, test_data: pd.DataFrame):
        """
        Function to sort the test data and set the target column to an initial value.

        Args:
            test_data (pd.DataFrame): The data the inferrence is desired on.

        Returns:
            The prepared test data
        """
        # Sort the data and group it by ID
        test_data = test_data.sort_values([self.asset_id_column, self.cycle_column])
        rul = pd.DataFrame(
            test_data.groupby(self.asset_id_column)[self.cycle_column].max()
        ).reset_index()

        # Set the RUL as in the training data
        # RUL = MaxCycle - CurrentCycle   although, that is incorrect for the test data
        rul.columns = [self.asset_id_column, "max"]
        test_data = test_data.merge(rul, on=[self.asset_id_column], how="left")

        test_data["RUL"] = test_data["max"] - test_data[self.cycle_column]
        test_data.drop("max", axis=1, inplace=True)

        test_data[self.target_column] = test_data["RUL"].apply(
            lambda x: self.max_RUL if x >= self.max_RUL else x
        )
        test_data.drop("RUL", axis=1, inplace=True)

        return test_data

    # This function normalizes the training or test dataset.
    def _data_scaler(self, data: pd.DataFrame):
        """
        function to standardize the dataset.
        The cycle column will be normalized between 0 and 1 to create the cycle_norm column.

        Parameters:
        - data (DataFrame): The data to be scaled.

        Returns:
            The scaled Dataframe.
        """
        min_max_scaler = MinMaxScaler()
        data[self.cycle_norm] = data[self.cycle_column]
        cols_normalize = data.columns.difference(
            [self.asset_id_column, self.cycle_column, self.target_column]
        )

        norm_df = pd.DataFrame(
            min_max_scaler.fit_transform(data[cols_normalize]),
            columns=cols_normalize,
            index=data.index,
        )
        join_df = data[data.columns.difference(cols_normalize)].join(norm_df)
        df = join_df.reindex(columns=data.columns)
        df = df.reset_index(drop=True)
        return df

    # Prepare the Test data for evaluation
    def _test_data_eval_prep(self, test_df, sequence_length, sequence_cols) -> None:
        """
        This function picks the last sequence for each
        engine in the test data for model evaluation

         Parameters:
        - test_df (DataFrame): The data to be prepared.
        - sequence_length (int): The number of time windows
        - sequence_cols (list): The columns in the data that will be used by the model

        Returns:
        None.
        """
        seq_array_test_last = [
            test_df[test_df[self.asset_id_column] == id][sequence_cols].values[
                -sequence_length:
            ]
            for id in test_df[self.asset_id_column].unique()
            if len(test_df[test_df[self.asset_id_column] == id]) >= sequence_length
        ]

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

        y_mask = [
            len(test_df[test_df[self.asset_id_column] == id]) >= sequence_length
            for id in test_df[self.asset_id_column].unique()
        ]
        label_array_test_last = (
            test_df.groupby(self.asset_id_column)[self.target_column]
            .nth(-1)[y_mask]
            .values
        )
        label_array_test_last = label_array_test_last.reshape(
            label_array_test_last.shape[0], 1
        ).astype(np.float32)

        return seq_array_test_last, label_array_test_last

    # function to generate labels
    def _gen_labels(self, id_df, seq_length, label) -> None:
        """
        This function generates labels for an individual component

         Parameters:
        - id_df (DataFrame): The data for just one ID.
        - seq_length (int): The number of time windows
        - label (list): The column(s) that labels will be generated for

        Returns:
        None.
        """
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]

    def _label_generator(
        self, train_data, sequence_length, sequence_cols
    ) -> np.ndarray:
        """generate labels

        Parameters:
        - train_data (DataFrame): The dataframe containing data to be labeled.
        - seq_length (int): The number of time windows
        - label (list): The column(s) that labels will be generated for

        Returns:
        The label array
        """
        label_gen = [
            self._gen_labels(
                train_data[train_data[self.asset_id_column] == id],
                sequence_length,
                [self.target_column],
            )
            for id in train_data[self.asset_id_column].unique()
        ]
        label_array = np.concatenate(label_gen).astype(np.float32)
        return label_array

    # These functions reshape the training and test datasets in three dimension
    def _gen_training_sequence(self, id_df, seq_length, seq_cols):
        """
        sequence generator

        Parameters:

            id_df: Engine ID
            seq_length: Sequence length
            seq_cols: Sequence column

        """

        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]

        for start, stop in zip(
            range(0, num_elements - seq_length), range(seq_length, num_elements)
        ):
            yield data_matrix[start:stop, :]

    def _sequence_training_gen(
        self, train_data, sequence_length, sequence_cols
    ) -> np.ndarray:
        """
        training sequence generator.

        Parameters:
        - train_data (DataFrame): The dataframe containing data to be labeled.
        - sequence_length (int): The number of time windows
        - sequence_cols (list): The columns of important data in the dataframe

        Returns:
            Generated sequence in the form
            [samples, time_steps, features]
        """
        seq_gen = (
            list(
                self._gen_training_sequence(
                    train_data[train_data[self.asset_id_column] == id],
                    sequence_length,
                    sequence_cols,
                )
            )
            for id in train_data[self.asset_id_column].unique()
        )

        # generate sequences and convert to numpy array
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        return seq_array

    def _build_model(self) -> None:
        """
        private function to instantiate the Attention_based multihead CNLSTM

        Parameters:
        None

        Returns:
        None - Internal field 'model' is instantiated
        """
        in_layers, out_layers = list(), list()
        for i in range(self.nb_features):
            inputs = Input(shape=(self.sequence_length, 1))
            rnn1 = Conv1D(
                filters=self.conv_filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )(inputs)
            lr1 = LeakyReLU()(rnn1)
            bn1 = BatchNormalization()(lr1)
            rnn2 = Conv1D(
                filters=self.conv_filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )(bn1)
            lr2 = LeakyReLU()(rnn2)
            bn2 = BatchNormalization()(lr2)
            rnn3 = LSTM(units=self.lstm_units, return_sequences=True)(bn2)
            lr3 = LeakyReLU()(rnn3)
            bn3 = BatchNormalization()(lr3)
            att = SeqSelfAttention(
                attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                kernel_regularizer=keras.regularizers.l2(self.regulizer_l2),
                bias_regularizer=keras.regularizers.l1(self.regulizer_l1),
                attention_regularizer_weight=self.attention_regulizer_weight,
                attention_width=self.attention_width,
            )(bn3)
            pool1 = MaxPooling1D(pool_size=2)(att)
            flat = Flatten()(pool1)
            # store layers
            in_layers.append(inputs)
            out_layers.append(flat)
        # merge heads
        merged = concatenate(out_layers)
        # interpretation
        dense1 = Dense(50, activation="relu")(merged)
        outputs = Dense(1)(dense1)  # Condense it down to one output feature
        self.model = Model(inputs=in_layers, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.optimizer_learning_rate)

        self.model.compile(
            loss=keras.losses.Huber(),
            optimizer=optimizer,
            metrics=[self._rmse, self._k_score],
        )

    def _confusion_matrix(self, y_true, y_pred):
        return cm(y_true, y_pred)

    def _k_score(self, y_true, y_pred):
        """score metric used for model evaluation"""
        d = y_pred - y_true
        return keras.backend.sum(
            keras.backend.exp(d[d >= 0] / 10) - 1
        ) + keras.backend.sum(keras.backend.exp(-1 * d[d < 0] / 13) - 1)

    def _rmse(self, y_true, y_pred):
        """rmse metric used for model evaluation"""
        return keras.backend.sqrt(
            keras.backend.mean(keras.backend.square(y_pred - y_true))
        )
