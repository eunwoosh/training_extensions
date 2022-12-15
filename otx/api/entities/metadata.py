"""This module defines classes representing metadata information."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import abc
from enum import Enum, auto
from typing import Optional

from otx.api.entities.id import ID
from otx.api.entities.model import ModelEntity


class IMetadata(metaclass=abc.ABCMeta):
    """This interface represents any additional metadata information which can be connected to an IMedia."""

    __name = Optional[str]

    @property
    def name(self):
        """Gets or sets the name of the Metadata item."""
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value


class FloatType(Enum):
    """Represents the use of the FloatMetadata."""

    FLOAT = auto()  # Regular float, without particular context
    EMBEDDING_VALUE = auto()
    ACTIVE_SCORE = auto()

    def __str__(self):
        """Return the name of FloatType enum."""
        return str(self.name)


class FloatMetadata(IMetadata):
    """This class represents metadata of type float.

    Args:
        name (str): Name of the metadata.
        value (float): Value of the metadata.
        float_type (FloatType): Type of the metadata.
    """

    def __init__(self, name: str, value: float, float_type: FloatType = FloatType.FLOAT):
        self.name = name
        self.value = value
        self.float_type = float_type

    def __repr__(self):
        """Prints the model, data and type of the MetadataItemEntity."""
        return f"FloatMetadata({self.name}, {self.value}, {self.float_type})"

    def __eq__(self, other):
        """Checks if two FloatMetadata have the same name, value and type."""
        return self.name == other.name and self.value == other.value and self.float_type == other.float_type


class VideoMetadata(IMetadata):
    """This class represents metadata of video.

    Args:
        video_id (ID): ID for video.
        frame_idx (int): Index for frame.
    """

    def __init__(self, video_id: ID, frame_idx: int):
        self.video_id = video_id
        self.frame_idx = frame_idx

    def __repr__(self):
        """Prints the video_id, frame_id and type of the MetadataItemEntity."""
        return f"VideoMetadata({self.video_id}, {self.frame_idx})"

    def __eq__(self, other):
        """Checks if two VideoMetadata have the same name, value and type."""
        return self.video_id == other.video_id and self.frame_idx == other.frame_idx


class VideoMetadata(IMetadata):
    """This class represents metadata of video.

    Args:
        video_id (ID): ID for video.
        frame_idx (int): Index for frame.
    """

    def __init__(self, name: str, video_id: int, frame_idx: int):
        self.name = name
        self.video_id = video_id
        self.frame_idx = frame_idx

    def __repr__(self):
        """Prints the name, video_id, frame_id and type of the MetadataItemEntity."""
        return f"VideoMetadata({self.name}, {self.video_id}, {self.frame_idx})"

    def __eq__(self, other):
        """Checks if two VideoMetadata have the same name, value and type."""
        return self.name == other.name and self.video_id == other.video_id and self.frame_idx == other.frame_idx


class MetadataItemEntity:
    """This class is a wrapper class which connects the metadata value to model, which was used to generate it.

    Args:
        data (IMetadata): The metadata value.
        model (Optional[ModelEntity]): The model which was used to generate the metadata. Defaults to None.
    """

    def __init__(
        self,
        data: IMetadata,
        model: Optional[ModelEntity] = None,
    ):
        self.data = data
        self.model = model

    def __repr__(self):
        """Prints the model and data of the MetadataItemEntity."""
        return f"MetadataItemEntity(model={self.model}, data={self.data})"

    def __eq__(self, other):
        """Returns true if the model and the data match the other MetadataItemEntity."""
        return self.model == other.model and self.data == other.data
