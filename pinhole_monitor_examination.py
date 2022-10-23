#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.4),
    on październik 23, 2022, at 17:55
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (
    NOT_STARTED,
    STARTED,
    PLAYING,
    PAUSED,
    STOPPED,
    FINISHED,
    PRESSED,
    RELEASED,
    FOREVER,
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (
    sin,
    cos,
    tan,
    log,
    log10,
    pi,
    average,
    sqrt,
    std,
    deg2rad,
    rad2deg,
    linspace,
    asarray,
)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = "2020.2.4"
expName = (
    "pinhole_monitor_examination"  # from the Builder filename that created this script
)
expInfo = {"participant": "", "session": "001"}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo["date"] = data.getDateStr()  # add a simple timestamp
expInfo["expName"] = expName
expInfo["psychopyVersion"] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = (
    _thisDir
    + os.sep
    + "data/%s_%s_%s" % (expInfo["participant"], expName, expInfo["date"])
)

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(
    name=expName,
    version="",
    extraInfo=expInfo,
    runtimeInfo=None,
    originPath="C:\\Users\\PK\\Desktop\\PsychoPy\\pinhole_monitor_examination.py",
    savePickle=True,
    saveWideText=True,
    dataFileName=filename,
)
# save a log file for detail verbose info
logFile = logging.LogFile(filename + ".log", level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[2560, 1600],
    fullscr=True,
    screen=1,
    winType="pyglet",
    allowGUI=False,
    allowStencil=False,
    monitor="testMonitor",
    color=[0, 0, 0],
    colorSpace="rgb",
    blendMode="avg",
    useFBO=True,
    units="height",
)
# store frame rate of monitor if we can measure it
expInfo["frameRate"] = win.getActualFrameRate()
if expInfo["frameRate"] != None:
    frameDur = 1.0 / round(expInfo["frameRate"])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "WelcomeScreen"
WelcomeScreenClock = core.Clock()
welcome_text = visual.TextStim(
    win=win,
    name="welcome_text",
    text="Start of the alignment",
    font="Arial",
    pos=(0, 0),
    height=0.1,
    wrapWidth=None,
    ori=0,
    color="white",
    colorSpace="rgb",
    opacity=1,
    languageStyle="LTR",
    depth=0.0,
)
key_resp = keyboard.Keyboard()

# Initialize components for Routine "aligment"
aligmentClock = core.Clock()
aligment_image = visual.ImageStim(
    win=win,
    name="aligment_image",
    image="sin",
    mask=None,
    ori=0,
    pos=(0, 0),
    size=None,
    color=[1, 1, 1],
    colorSpace="rgb",
    opacity=1,
    flipHoriz=False,
    flipVert=False,
    texRes=128,
    interpolate=True,
    depth=0.0,
)
aligment_keys = keyboard.Keyboard()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

p = 17  # pixels per pinhole
d = 0.055  # interpupilary distance
l = 1

Backward = 0.03
Front = -0.06
h = 0.005
OP = (Backward + Front) / 2
alignment = 0

control_b = 0
control_f = 0
checking_point = False
loop_end = False
trials_finised = False

recon_path = "C:/Users/PK/Desktop/PsychoPy/img"
resolution = 1600, 2560  # tablet resolution


def object_positions(z, y, h, g, p_s, p, resolution_0):
    # all data in meters except parameter p
    # z   - object position in z
    # y   - object position in y
    # h   - object thickness
    # g   - gap between tablet and pinhole array
    # p_s - pixel size
    # p   - pixels per pinhole

    if z == 0:
        z += 0.000001

    z /= p_s
    y /= p_s
    h /= p_s * 2
    g /= -p_s
    p /= 2

    number_of_iteration = resolution_0 / (p * 2)
    pinhole_position = -(resolution_0 / 2) - p

    positions = []
    for i in range(int(number_of_iteration)):
        pinhole_position += 2 * p
        y_position_1 = ((y + h - pinhole_position) * g / z) + pinhole_position
        y_position_2 = ((y - h - pinhole_position) * g / z) + pinhole_position

        if y_position_1 < pinhole_position - p:
            y_position_1 = pinhole_position - p
        elif y_position_1 > pinhole_position + p:
            y_position_1 = pinhole_position + p

        if y_position_2 < pinhole_position - p:
            y_position_2 = pinhole_position - p
        elif y_position_2 > pinhole_position + p:
            y_position_2 = pinhole_position + p

        y_position_1 = 2 * pinhole_position - y_position_1
        y_position_2 = 2 * pinhole_position - y_position_2

        y_position_1 += number_of_iteration * p
        y_position_2 += number_of_iteration * p

        y_position_1 = int(y_position_1)
        y_position_2 = int(y_position_2)

        positions.append(sorted((y_position_1, y_position_2)))

    return positions


def object_width(object_positions, resolution):
    maximium = 0
    minimum = resolution[1]
    for positions in object_positions:
        if positions[0] != positions[1]:
            if max(positions) > maximium:
                maximium = max(positions)
            if min(positions) < minimum:
                minimum = min(positions)

    return minimum, maximium, maximium - minimum


def image_reconstruction(z, y, h, g, p_s, p, resolution, alignment, recon_path):
    object_positions_1 = object_positions(0, -y, h, g, p_s, p, resolution[1])
    object_positions_2 = object_positions(z, y, h, g, p_s, p, resolution[1])

    image = np.zeros(resolution)

    for positions in object_positions_1:
        image[:, positions[0] : positions[1]] = 255

    for positions in object_positions_2:
        image[:, positions[0] : positions[1]] = 255

    object_width_1 = object_width(object_positions_1, resolution)
    object_width_2 = object_width(object_positions_2, resolution)
    edge = int((object_width_2[2] - object_width_1[2]) / 2)

    if z > 0:
        slope = 1
        intensity = edge
    elif edge > 0:
        slope = int(255 / edge)
        intensity = 255

    if edge != 0:
        for j in range(edge):
            intensity -= slope
            if intensity < 0:
                intensity = 0
            image[:, object_width_1[0] - j - 1 : object_width_1[0] - j] = intensity
            image[:, object_width_1[1] + j : object_width_1[1] + j + 1] = intensity

    ali_array = np.zeros((resolution[0], abs(alignment)))

    if alignment > 0:
        image = np.concatenate((ali_array, image), axis=1)
        image = image[:, :-alignment]
    else:
        image = np.concatenate((image, ali_array), axis=1)
        image = image[:, abs(alignment) :]

    final = Image.fromarray(np.uint8(image))
    final = final.convert("RGB")
    final.save(f"{recon_path}" + f"/IMG.jpg")


image_reconstruction(0.025, 0.01, h, 0.004, 71e-6, p, resolution, alignment, recon_path)

# Initialize components for Routine "Middle_screen"
Middle_screenClock = core.Clock()
middle_text = visual.TextStim(
    win=win,
    name="middle_text",
    text="Start of the examination",
    font="Arial",
    pos=(0, 0),
    height=0.1,
    wrapWidth=None,
    ori=0,
    color="white",
    colorSpace="rgb",
    opacity=1,
    languageStyle="LTR",
    depth=0.0,
)
middle_key = keyboard.Keyboard()

# Initialize components for Routine "random_choice"
random_choiceClock = core.Clock()
random_choice_image = visual.ImageStim(
    win=win,
    name="random_choice_image",
    image="sin",
    mask=None,
    ori=0,
    pos=(0, 0),
    size=None,
    color=[1, 1, 1],
    colorSpace="rgb",
    opacity=1,
    flipHoriz=False,
    flipVert=False,
    texRes=128,
    interpolate=True,
    depth=0.0,
)
random_choice_keys = keyboard.Keyboard()

# Initialize components for Routine "code_2"
code_2Clock = core.Clock()
image_1 = visual.ImageStim(
    win=win,
    name="image_1",
    image="sin",
    mask=None,
    ori=0,
    pos=(0, 0),
    size=None,
    color=[1, 1, 1],
    colorSpace="rgb",
    opacity=1,
    flipHoriz=False,
    flipVert=False,
    texRes=128,
    interpolate=True,
    depth=0.0,
)
key_response = keyboard.Keyboard()

# Initialize components for Routine "EndScreen"
EndScreenClock = core.Clock()
text = visual.TextStim(
    win=win,
    name="text",
    text="End of the examination",
    font="Arial",
    pos=(0, 0),
    height=0.1,
    wrapWidth=None,
    ori=0,
    color="white",
    colorSpace="rgb",
    opacity=1,
    languageStyle="LTR",
    depth=0.0,
)
key_resp_2 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = (
    core.CountdownTimer()
)  # to track time remaining of each (non-slip) routine

# ------Prepare to start Routine "WelcomeScreen"-------
continueRoutine = True
# update component parameters for each repeat
key_resp.keys = []
key_resp.rt = []
_key_resp_allKeys = []
# keep track of which components have finished
WelcomeScreenComponents = [welcome_text, key_resp]
for thisComponent in WelcomeScreenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
WelcomeScreenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "WelcomeScreen"-------
while continueRoutine:
    # get current time
    t = WelcomeScreenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=WelcomeScreenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *welcome_text* updates
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, "tStartRefresh")  # time at next scr refresh
        welcome_text.setAutoDraw(True)

    # *key_resp* updates
    waitOnFlip = False
    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        key_resp.frameNStart = frameN  # exact frame index
        key_resp.tStart = t  # local t and not account for scr refresh
        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp, "tStartRefresh")  # time at next scr refresh
        key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(
            key_resp.clearEvents, eventType="keyboard"
        )  # clear events on next screen flip
    if key_resp.status == STARTED and not waitOnFlip:
        theseKeys = key_resp.getKeys(keyList=["space"], waitRelease=False)
        _key_resp_allKeys.extend(theseKeys)
        if len(_key_resp_allKeys):
            key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
            key_resp.rt = _key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = (
        False  # will revert to True if at least one component still running
    )
    for thisComponent in WelcomeScreenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if (
        continueRoutine
    ):  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "WelcomeScreen"-------
for thisComponent in WelcomeScreenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData("welcome_text.started", welcome_text.tStartRefresh)
thisExp.addData("welcome_text.stopped", welcome_text.tStopRefresh)
# check responses
if key_resp.keys in ["", [], None]:  # No response was made
    key_resp.keys = None
thisExp.addData("key_resp.keys", key_resp.keys)
if key_resp.keys != None:  # we had a response
    thisExp.addData("key_resp.rt", key_resp.rt)
thisExp.addData("key_resp.started", key_resp.tStartRefresh)
thisExp.addData("key_resp.stopped", key_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
aligment_loop = data.TrialHandler(
    nReps=9999,
    method="sequential",
    extraInfo=expInfo,
    originPath=-1,
    trialList=[None],
    seed=None,
    name="aligment_loop",
)
thisExp.addLoop(aligment_loop)  # add the loop to the experiment
thisAligment_loop = aligment_loop.trialList[
    0
]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisAligment_loop.rgb)
if thisAligment_loop != None:
    for paramName in thisAligment_loop:
        exec("{} = thisAligment_loop[paramName]".format(paramName))

for thisAligment_loop in aligment_loop:
    currentLoop = aligment_loop
    # abbreviate parameter names if possible (e.g. rgb = thisAligment_loop.rgb)
    if thisAligment_loop != None:
        for paramName in thisAligment_loop:
            exec("{} = thisAligment_loop[paramName]".format(paramName))

    # ------Prepare to start Routine "aligment"-------
    continueRoutine = True
    # update component parameters for each repeat
    aligment_image.setImage("img/IMG.jpg")
    aligment_keys.keys = []
    aligment_keys.rt = []
    _aligment_keys_allKeys = []
    # keep track of which components have finished
    aligmentComponents = [aligment_image, aligment_keys]
    for thisComponent in aligmentComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    aligmentClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1

    # -------Run Routine "aligment"-------
    while continueRoutine:
        # get current time
        t = aligmentClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=aligmentClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *aligment_image* updates
        if aligment_image.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            aligment_image.frameNStart = frameN  # exact frame index
            aligment_image.tStart = t  # local t and not account for scr refresh
            aligment_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(aligment_image, "tStartRefresh")  # time at next scr refresh
            aligment_image.setAutoDraw(True)

        # *aligment_keys* updates
        waitOnFlip = False
        if aligment_keys.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            aligment_keys.frameNStart = frameN  # exact frame index
            aligment_keys.tStart = t  # local t and not account for scr refresh
            aligment_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(aligment_keys, "tStartRefresh")  # time at next scr refresh
            aligment_keys.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(aligment_keys.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(
                aligment_keys.clearEvents, eventType="keyboard"
            )  # clear events on next screen flip
        if aligment_keys.status == STARTED and not waitOnFlip:
            theseKeys = aligment_keys.getKeys(
                keyList=["left", "right", "space"], waitRelease=False
            )
            _aligment_keys_allKeys.extend(theseKeys)
            if len(_aligment_keys_allKeys):
                aligment_keys.keys = [
                    key.name for key in _aligment_keys_allKeys
                ]  # storing all keys
                aligment_keys.rt = [key.rt for key in _aligment_keys_allKeys]
                # a response ends the routine
                continueRoutine = False

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = (
            False  # will revert to True if at least one component still running
        )
        for thisComponent in aligmentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if (
            continueRoutine
        ):  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # -------Ending Routine "aligment"-------
    for thisComponent in aligmentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    aligment_loop.addData("aligment_image.started", aligment_image.tStartRefresh)
    aligment_loop.addData("aligment_image.stopped", aligment_image.tStopRefresh)
    # check responses
    if aligment_keys.keys in ["", [], None]:  # No response was made
        aligment_keys.keys = None
    aligment_loop.addData("aligment_keys.keys", aligment_keys.keys)
    if aligment_keys.keys != None:  # we had a response
        aligment_loop.addData("aligment_keys.rt", aligment_keys.rt)
    aligment_loop.addData("aligment_keys.started", aligment_keys.tStartRefresh)
    aligment_loop.addData("aligment_keys.stopped", aligment_keys.tStopRefresh)
    if "left" in aligment_keys.keys[-1]:
        alignment -= 1
    elif "right" in aligment_keys.keys[-1]:
        alignment += 1
    elif "space" in aligment_keys.keys[-1]:
        aligment_loop.finished = True

    aligment_keys.keys.pop()
    image_reconstruction(
        0.025, 0.01, h, 0.004, 71e-6, p, resolution, alignment, recon_path
    )

    if aligment_loop.finished:
        image_reconstruction(
            0, 0.01, h, 0.004, 71e-6, p, resolution, alignment, recon_path
        )
    # the Routine "aligment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()

# completed 9999 repeats of 'aligment_loop'


# ------Prepare to start Routine "Middle_screen"-------
continueRoutine = True
# update component parameters for each repeat
middle_key.keys = []
middle_key.rt = []
_middle_key_allKeys = []
# keep track of which components have finished
Middle_screenComponents = [middle_text, middle_key]
for thisComponent in Middle_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Middle_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Middle_screen"-------
while continueRoutine:
    # get current time
    t = Middle_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Middle_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *middle_text* updates
    if middle_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        middle_text.frameNStart = frameN  # exact frame index
        middle_text.tStart = t  # local t and not account for scr refresh
        middle_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(middle_text, "tStartRefresh")  # time at next scr refresh
        middle_text.setAutoDraw(True)

    # *middle_key* updates
    waitOnFlip = False
    if middle_key.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        middle_key.frameNStart = frameN  # exact frame index
        middle_key.tStart = t  # local t and not account for scr refresh
        middle_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(middle_key, "tStartRefresh")  # time at next scr refresh
        middle_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(middle_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(
            middle_key.clearEvents, eventType="keyboard"
        )  # clear events on next screen flip
    if middle_key.status == STARTED and not waitOnFlip:
        theseKeys = middle_key.getKeys(keyList=["space"], waitRelease=False)
        _middle_key_allKeys.extend(theseKeys)
        if len(_middle_key_allKeys):
            middle_key.keys = _middle_key_allKeys[-1].name  # just the last key pressed
            middle_key.rt = _middle_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = (
        False  # will revert to True if at least one component still running
    )
    for thisComponent in Middle_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if (
        continueRoutine
    ):  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Middle_screen"-------
for thisComponent in Middle_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData("middle_text.started", middle_text.tStartRefresh)
thisExp.addData("middle_text.stopped", middle_text.tStopRefresh)
# check responses
if middle_key.keys in ["", [], None]:  # No response was made
    middle_key.keys = None
thisExp.addData("middle_key.keys", middle_key.keys)
if middle_key.keys != None:  # we had a response
    thisExp.addData("middle_key.rt", middle_key.rt)
thisExp.addData("middle_key.started", middle_key.tStartRefresh)
thisExp.addData("middle_key.stopped", middle_key.tStopRefresh)
thisExp.nextEntry()
# the Routine "Middle_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "random_choice"-------
continueRoutine = True
# update component parameters for each repeat
random_choice_image.setImage("img/IMG.jpg")
random_choice_keys.keys = []
random_choice_keys.rt = []
_random_choice_keys_allKeys = []
# keep track of which components have finished
random_choiceComponents = [random_choice_image, random_choice_keys]
for thisComponent in random_choiceComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
random_choiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "random_choice"-------
while continueRoutine:
    # get current time
    t = random_choiceClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=random_choiceClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *random_choice_image* updates
    if random_choice_image.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        random_choice_image.frameNStart = frameN  # exact frame index
        random_choice_image.tStart = t  # local t and not account for scr refresh
        random_choice_image.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(random_choice_image, "tStartRefresh")  # time at next scr refresh
        random_choice_image.setAutoDraw(True)

    # *random_choice_keys* updates
    waitOnFlip = False
    if random_choice_keys.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        random_choice_keys.frameNStart = frameN  # exact frame index
        random_choice_keys.tStart = t  # local t and not account for scr refresh
        random_choice_keys.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(random_choice_keys, "tStartRefresh")  # time at next scr refresh
        random_choice_keys.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(random_choice_keys.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(
            random_choice_keys.clearEvents, eventType="keyboard"
        )  # clear events on next screen flip
    if random_choice_keys.status == STARTED and not waitOnFlip:
        theseKeys = random_choice_keys.getKeys(keyList=["f", "b"], waitRelease=False)
        _random_choice_keys_allKeys.extend(theseKeys)
        if len(_random_choice_keys_allKeys):
            random_choice_keys.keys = _random_choice_keys_allKeys[
                -1
            ].name  # just the last key pressed
            random_choice_keys.rt = _random_choice_keys_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = (
        False  # will revert to True if at least one component still running
    )
    for thisComponent in random_choiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if (
        continueRoutine
    ):  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "random_choice"-------
for thisComponent in random_choiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData("random_choice_image.started", random_choice_image.tStartRefresh)
thisExp.addData("random_choice_image.stopped", random_choice_image.tStopRefresh)
# check responses
if random_choice_keys.keys in ["", [], None]:  # No response was made
    random_choice_keys.keys = None
thisExp.addData("random_choice_keys.keys", random_choice_keys.keys)
if random_choice_keys.keys != None:  # we had a response
    thisExp.addData("random_choice_keys.rt", random_choice_keys.rt)
thisExp.addData("random_choice_keys.started", random_choice_keys.tStartRefresh)
thisExp.addData("random_choice_keys.stopped", random_choice_keys.tStopRefresh)
thisExp.nextEntry()
if "f" in random_choice_keys.keys[-1]:
    BP = Backward
    FP = Front
elif "b" in random_choice_keys.keys[-1]:
    BP = -Front
    FP = -Backward

OP = (BP + FP) / 2
image_reconstruction(OP, 0.01, h, 0.004, 71e-6, p, resolution, alignment, recon_path)

obj_positions = []
obj_positions.append([BP, OP, FP])

plt.ion()
fig, ax = plt.subplots()
figManager = plt.get_current_fig_manager()
figManager.window.move(0, 0)
point_position = 0
plot = ax.scatter(
    [point_position, point_position, point_position],
    obj_positions[0],
    c=["b", "r", "b"],
)
ax.set_ylim((FP - 0.01, BP + 0.01))
ax.hlines(y=0, xmin=-1, xmax=point_position + 1)
ax.vlines(x=point_position, ymin=obj_positions[-1][2], ymax=obj_positions[-1][0])

# the Routine "random_choice" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(
    nReps=9999,
    method="sequential",
    extraInfo=expInfo,
    originPath=-1,
    trialList=[None],
    seed=None,
    name="trials",
)
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec("{} = thisTrial[paramName]".format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec("{} = thisTrial[paramName]".format(paramName))

    # ------Prepare to start Routine "code_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    image_1.setImage("img/IMG.jpg")
    key_response.keys = []
    key_response.rt = []
    _key_response_allKeys = []
    # keep track of which components have finished
    code_2Components = [image_1, key_response]
    for thisComponent in code_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, "status"):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    code_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1

    # -------Run Routine "code_2"-------
    while continueRoutine:
        # get current time
        t = code_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=code_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *image_1* updates
        if image_1.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            image_1.frameNStart = frameN  # exact frame index
            image_1.tStart = t  # local t and not account for scr refresh
            image_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_1, "tStartRefresh")  # time at next scr refresh
            image_1.setAutoDraw(True)

        # *key_response* updates
        waitOnFlip = False
        if key_response.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            key_response.frameNStart = frameN  # exact frame index
            key_response.tStart = t  # local t and not account for scr refresh
            key_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_response, "tStartRefresh")  # time at next scr refresh
            key_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(
                key_response.clearEvents, eventType="keyboard"
            )  # clear events on next screen flip
        if key_response.status == STARTED and not waitOnFlip:
            theseKeys = key_response.getKeys(keyList=["b", "f"], waitRelease=False)
            _key_response_allKeys.extend(theseKeys)
            if len(_key_response_allKeys):
                key_response.keys = [
                    key.name for key in _key_response_allKeys
                ]  # storing all keys
                key_response.rt = [key.rt for key in _key_response_allKeys]
                # a response ends the routine
                continueRoutine = False

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = (
            False  # will revert to True if at least one component still running
        )
        for thisComponent in code_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if (
            continueRoutine
        ):  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # -------Ending Routine "code_2"-------
    for thisComponent in code_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData("image_1.started", image_1.tStartRefresh)
    trials.addData("image_1.stopped", image_1.tStopRefresh)
    # check responses
    if key_response.keys in ["", [], None]:  # No response was made
        key_response.keys = None
    trials.addData("key_response.keys", key_response.keys)
    if key_response.keys != None:  # we had a response
        trials.addData("key_response.rt", key_response.rt)
    trials.addData("key_response.started", key_response.tStartRefresh)
    trials.addData("key_response.stopped", key_response.tStopRefresh)
    # Adding choosed button to results list
    letter = key_response.keys[-1]
    obj_positions[-1].append(letter)

    # Adding data to excel file
    trials.addData("Front_Boundary", FP)
    trials.addData("Backward_Boundary", BP)
    trials.addData("Object_Position", OP)
    trials.addData("Percent", (BP - FP) / (Backward - Front))
    trials.addData("Choosed_Position", letter)
    trials.addData("Omega", np.rad2deg(abs(d * OP) / l ** 2) * 60)

    # Handling normal situation
    if control_f != 2 or control_b != 2:
        if "b" in key_response.keys[-1]:
            BP = OP
            control_b += 1
            control_f = 0
        elif "f" in key_response.keys[-1]:
            FP = OP
            control_f += 1
            control_b = 0

    # Handling checking point after two same observer responses
    if checking_point:
        # Handling situation when checking point occurs before three steps
        if "f" in key_response.keys[-1] and OP < 0:
            check = True
        elif "b" in key_response.keys[-1] and OP > 0:
            check = True
        else:
            check = False

        # Handling situation when checking point occurs after two steps
        for point in reversed(obj_positions[:-2]):
            if point[0] != point[2]:
                if point[1] == OP and point[3] == letter:
                    check = True
                    break
                elif point[1] == OP and point[3] != letter:
                    check = False
                    break

        # Points assigment
        if check:
            if obj_positions[-2][3] == "f":
                BP = obj_positions[-2][0]
                FP = obj_positions[-2][1]
            else:
                BP = obj_positions[-2][1]
                FP = obj_positions[-2][2]
        else:
            BP = obj_positions[-3][0]
            FP = obj_positions[-3][2]

        control_f = 0
        control_b = 0
        checking_point = False

    # Adding checking point indicator
    if control_f == 2:
        FP = BP
        checking_point = True
    elif control_b == 2:
        BP = FP
        checking_point = True

    # Adding end loop indicator if there are two the same checking points in a row
    if len(obj_positions) >= 6:
        if (
            obj_positions[-1][1] == obj_positions[-4][1]
            and obj_positions[-1][3] == obj_positions[-4][3]
            and obj_positions[-1][0] == obj_positions[-1][2]
        ):
            loop_end = True

    # Points calculation and drawing plot
    OP = (FP + BP) / 2
    image_reconstruction(
        OP, 0.01, h, 0.004, 71e-6, p, resolution, alignment, recon_path
    )
    point_position += 1
    obj_positions.append([BP, OP, FP])
    key_response.keys.pop()

    if obj_positions[-1][0] == obj_positions[-1][1]:
        ax.scatter(
            [point_position, point_position, point_position],
            obj_positions[-1],
            c=["r", "r", "r"],
        )
    else:
        ax.scatter(
            [point_position, point_position, point_position],
            obj_positions[-1],
            c=["b", "r", "b"],
        )
        ax.vlines(
            x=point_position, ymin=obj_positions[-1][2], ymax=obj_positions[-1][0]
        )

    ax.text(x=point_position - 1, y=obj_positions[-2][1], s=letter, fontsize=20)
    ax.hlines(y=0, xmin=-1, xmax=point_position + 1)
    # fig.canvas.draw()

    # Trial ending
    if trials_finised:
        trials.finished = True

    # Loop ending if conditions were met
    if BP != FP and ((BP - FP) / (Backward - Front)) < 0.05:
        plt.savefig(filename + ".jpg")
        trials.finished = True
    elif loop_end:
        plt.savefig(filename + ".jpg")
        trials_finised = True

    # the Routine "code_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()

# completed 9999 repeats of 'trials'


# ------Prepare to start Routine "EndScreen"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_2.keys = []
key_resp_2.rt = []
_key_resp_2_allKeys = []
# keep track of which components have finished
EndScreenComponents = [text, key_resp_2]
for thisComponent in EndScreenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
EndScreenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "EndScreen"-------
while continueRoutine:
    # get current time
    t = EndScreenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=EndScreenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, "tStartRefresh")  # time at next scr refresh
        text.setAutoDraw(True)

    # *key_resp_2* updates
    waitOnFlip = False
    if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.tStart = t  # local t and not account for scr refresh
        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_2, "tStartRefresh")  # time at next scr refresh
        key_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(
            key_resp_2.clearEvents, eventType="keyboard"
        )  # clear events on next screen flip
    if key_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_2.getKeys(keyList=["space"], waitRelease=False)
        _key_resp_2_allKeys.extend(theseKeys)
        if len(_key_resp_2_allKeys):
            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = (
        False  # will revert to True if at least one component still running
    )
    for thisComponent in EndScreenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if (
        continueRoutine
    ):  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "EndScreen"-------
for thisComponent in EndScreenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData("text.started", text.tStartRefresh)
thisExp.addData("text.stopped", text.tStopRefresh)
# check responses
if key_resp_2.keys in ["", [], None]:  # No response was made
    key_resp_2.keys = None
thisExp.addData("key_resp_2.keys", key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData("key_resp_2.rt", key_resp_2.rt)
thisExp.addData("key_resp_2.started", key_resp_2.tStartRefresh)
thisExp.addData("key_resp_2.stopped", key_resp_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "EndScreen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
thisExp.addData("Front_Boundary", FP)
thisExp.addData("Backward_Boundary", BP)
thisExp.addData("Object_Position", OP)
thisExp.addData("Percent", (BP - FP) / (Backward - Front))
thisExp.addData("Omega", np.rad2deg(abs(d * OP) / l ** 2) * 60)

# Flip one final time so any remaining win.callOnFlip()
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename + ".csv", delim="auto")
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
