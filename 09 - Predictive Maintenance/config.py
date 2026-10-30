from pydantic import BaseModel, Field


class RULDefaults:
    LSTM_UNITS: int = 64
    CONV_FILTERS: int = 64
    KERNEL_SIZE: int = 3
    TRAINING_EPOCHS: int = 100
    TRAINING_PATIENCE: int = 5
    VAL_SPLIT: float = 0.1
    ATTENTION_WIDTH = 15
    REGULIZER_L1 = 1e-4
    REGULIZER_L2 = 1e-4
    ATT_REG_WEIGHT = 1e-4
    OPTIMIZER_LEARNING_RATE = 1e-5
    SEQ_LENGTH: int = 30
    ASSET_ID_COL: str = "id"
    CYCLE_COL: str = "cycle"
    MAX_RUL: int = 130
    TARGET_COLUMN: str = "R_early"
    SAVE_FILE: str = "../models/tempModel.h5"


class RULConfig(BaseModel):
    lstm_units: int = Field(
        description="Parameter to the LSTM Layer.  Positive integer, dimensionality of the output space.",
        default=RULDefaults.LSTM_UNITS,
    )
    conv_filters: int = Field(
        description=(
            "Parameter to the Convolutional Layers.  The dimension of the output space, number of "
            "filters in the convolution."
        ),
        default=RULDefaults.CONV_FILTERS,
    )
    kernel_size: int = Field(
        description="Parameter to the Convolutional Layers.  Specifies the size of the convolutional window.",
        default=RULDefaults.KERNEL_SIZE,
    )
    training_epochs: int = Field(
        description="When training the model, this is the number of epochs to train over",
        default=RULDefaults.TRAINING_EPOCHS,
    )
    training_patience: int = Field(
        description="When training the model, stop training early if there is no improvement after this many epochs",
        default=RULDefaults.TRAINING_PATIENCE,
    )

    validation_split: float = Field(
        description=("Percentage of training data used for validation."),
        default=RULDefaults.VAL_SPLIT,
    )

    attention_width: int = Field(
        description=(
            "Width of the local attention of the SeqSelfAttention portion of the model"
        ),
        default=RULDefaults.ATTENTION_WIDTH,
    )

    regulizer_l1: float = Field(
        description=("A regularizer that applies a L1 regularization penalty."),
        default=RULDefaults.REGULIZER_L1,
    )

    regulizer_l2: float = Field(
        description=("A regularizer that applies a L2 regularization penalty."),
        default=RULDefaults.REGULIZER_L2,
    )

    attention_regulizer_weight: float = Field(
        description=("The weights of attention regularizer."),
        default=RULDefaults.ATT_REG_WEIGHT,
    )

    optimizer_learning_rate: float = Field(
        description=("Learning rate for the optimizer"),
        default=RULDefaults.OPTIMIZER_LEARNING_RATE,
    )

    seq_length: int = Field(
        description=(
            "Parameter to the LSTM.  Size of the time windows ranging from 20-120. A danger of going too high "
            "on this is if the number of cycles given for an inference is too small"
        ),
        le=120,
        ge=20,
        default=RULDefaults.SEQ_LENGTH,
    )

    asset_id_column: str = Field(
        description=(
            "Name of the column in the input data that represents the unique identifier of the asset."
        ),
        default=RULDefaults.ASSET_ID_COL,
    )
    cycle_column: str = Field(
        description=(
            "Name of the column in the input data that represents the cycle column"
            " The cycle is the time reference for this type of data.  For each cycle,"
            " there is a set of sensor measurements.  The RUL predictions this model makes"
            " are a prediction of how many cycles the asset has before maintenance is required"
        ),
        default=RULDefaults.CYCLE_COL,
    )
    sensor_columns: list[str] = Field(
        description=(
            "Sensor data will be used by the model to determine the remaining "
            "useful life of the asset in question."
        ),
    )

    max_RUL: int = Field(
        description=(
            "This value is the Maximum that a RUL should be.  In production, it would be the "
            "maximum cycles to the next schedule maintenance or the minimum life of new assets of this family "
            "put into production."
        ),
        default=RULDefaults.MAX_RUL,
    )

    target_column: str = Field(
        description=(
            "Name of the column in the input data that represents the target column."
        ),
        default=RULDefaults.TARGET_COLUMN,
    )

    save_file_name: str = Field(
        description=("Name of the file the trained model will be saved into."),
        default=RULDefaults.SAVE_FILE,
    )
