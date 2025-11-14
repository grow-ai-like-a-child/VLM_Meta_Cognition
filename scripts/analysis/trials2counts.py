#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[translated by alan lee from trials2counts.m by Maniscalco & Lau (2012)]
[requires numpy-1.16.4, scipy-1.3.0, or later versions]
[comments below are copied and pasted from trials2counts.m]

function [nR_S1, nR_S2] = trials2counts(stimID, response, rating, nRatings, padCells, padAmount)

% Given data from an experiment where an observer discriminates between two
% stimulus alternatives on every trial and provides confidence ratings,
% converts trial by trial experimental information for N trials into response 
% counts.
%
% INPUTS
% stimID:   1xN vector. stimID(i) = 0 --> stimulus on i'th trial was S1.
%                       stimID(i) = 1 --> stimulus on i'th trial was S2.
%
% response: 1xN vector. response(i) = 0 --> response on i'th trial was "S1".
%                       response(i) = 1 --> response on i'th trial was "S2".
%
% rating:   1xN vector. rating(i) = X --> rating on i'th trial was X.
%                       X must be in the range 1 <= X <= nRatings.
%
% N.B. all trials where stimID is not 0 or 1, response is not 0 or 1, or
% rating is not in the range [1, nRatings], are omitted from the response
% count.
%
% nRatings: total # of available subjective ratings available for the
%           subject. e.g. if subject can rate confidence on a scale of 1-4,
%           then nRatings = 4
%
% optional inputs
%
% padCells: if set to 1, each response count in the output has the value of
%           padAmount added to it. Padding cells is desirable if trial counts 
%           of 0 interfere with model fitting.
%           if set to 0, trial counts are not manipulated and 0s may be
%           present in the response count output.
%           default value for padCells is 0.
%
% padAmount: the value to add to each response count if padCells is set to 1.
%            default value is 1/(2*nRatings)
%
%
% OUTPUTS
% nR_S1, nR_S2
% these are vectors containing the total number of responses in
% each response category, conditional on presentation of S1 and S2.
%
% e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 100 times
% responded S1, rating=2 : 50 times
% responded S1, rating=1 : 20 times
% responded S2, rating=1 : 10 times
% responded S2, rating=2 : 5 times
% responded S2, rating=3 : 1 time
%
% The ordering of response / rating counts for S2 should be the same as it
% is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 3 times
% responded S1, rating=2 : 7 times
% responded S1, rating=1 : 8 times
% responded S2, rating=1 : 12 times
% responded S2, rating=2 : 27 times
% responded S2, rating=3 : 89 times
"""

def trials2counts(stimID, response, rating, nRatings, padCells = 0, padAmount = None):

    ''' sort inputs '''
    # check for valid inputs
    if not ( len(stimID) == len(response)) and (len(stimID) == len(rating)):
        raise('stimID, response, and rating input vectors must have the same lengths')
    
    ''' filter bad trials '''
    tempstim = []
    tempresp = []
    tempratg = []
    for s,rp,rt in zip(stimID,response,rating):
        if (s == 0 or s == 1) and (rp == 0 or rp == 1) and (rt >=1 and rt <= nRatings):
            tempstim.append(s)
            tempresp.append(rp)
            tempratg.append(rt)
    stimID = tempstim
    response = tempresp
    rating = tempratg
    
    ''' set input defaults '''
    if padAmount == None:
        padAmount = 1/(2*nRatings)
    
    ''' compute response counts '''
    nR_S1 = []
    nR_S2 = []

    # S1 responses
    for r in range(nRatings,0,-1):
        cs1, cs2 = 0,0
        for s,rp,rt in zip(stimID, response, rating):
            if s==0 and rp==0 and rt==r:
                cs1 += 1
            if s==1 and rp==0 and rt==r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)
    
    # S2 responses
    for r in range(1,nRatings+1,1):
        cs1, cs2 = 0,0
        for s,rp,rt in zip(stimID, response, rating):
            if s==0 and rp==1 and rt==r:
                cs1 += 1
            if s==1 and rp==1 and rt==r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)
    
    # pad response counts to avoid zeros
    if padCells:
        nR_S1 = [n+padAmount for n in nR_S1]
        nR_S2 = [n+padAmount for n in nR_S2]
        
    return nR_S1, nR_S2

''' try running '''
'''
stimID =    [0, 1, 0, 0, 1, 1, 1, 1]
response =  [0, 1, 1, 1, 0, 0, 1, 1]
rating =    [1, 2, 3, 4, 4, 3, 2, 1]
nRatings = 4

nR_S1, nR_S2 = trials2counts(stimID, response, rating, nRatings,1)
print(nR_S1, nR_S2)
'''