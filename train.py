#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import json
import sys
#Recent import 21/9/18
import tensorflowjs as tfjs
#######################
from keras.models import Sequential 
from keras.layers import *
from keras.optimizers import *
from keras.layers import LSTM
from keras.layers import Dropout

from keras.layers import Input
import numpy as np

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: python train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=1000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    

    model = Sequential()
    print('env' , (10,) + (num_last_frames, ) +  env.observation_shape )
    #Printing the model params 

    #Output => env (4,) ObservationSHape (10, 10)

    # Convolutions.()
    model.add(TimeDistributed(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first') ,
        input_shape= (10 , ) + (num_last_frames, ) +  env.observation_shape
    ))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    )))
    model.add(TimeDistributed(Activation('relu')))
    #model.add(TimeDistributed(Conv2D(96, (11, 11),strides = (4,4), activation = 'relu')
        #, input_shape=(10, 225, 225, 3)))
    # Dense layers.
    model.add(TimeDistributed(Flatten()))
    #Timedistributed changes and the below 1 liner code change by me.
    model.add(Dropout(0.25))
    ##Adding Lstm layer to the model MY CHANGE on 24/12/2018.
    model.add(LSTM(units = 200 , return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 200 , return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 200))
    #My code change END#####
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))
    model.summary()
    model.compile(RMSprop(), 'MSE')
    #Recent code change -> 21/9/18
    model.save("Keras-64x2-10epoch")
    #tfjs.converters.save_keras_model(model, "tfjsv3")
    ##############################
    return model


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    model = create_dqn_model(env, num_last_frames=4)

    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=model.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95
    )


if __name__ == '__main__':
    main()
