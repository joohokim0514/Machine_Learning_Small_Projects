import argparse
import enum
import pickle
import sys
import os
# from utils import *
from collections import deque

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', help='primary algorithm to use')
parser.add_argument('--params', help='parameters accompanying the algorithm')


def stableMarriage(malePreference, femalePreference):
    advisorAssignment = {} # male id to female
    studentAssignment = {} # female id to male
    # compare the two dictionaries for a sanity check

    '''IMPLEMENT METHOD HERE'''
    unmatched = deque()
    for m in list(malePreference.keys()):
        unmatched.append(m)
        advisorAssignment[m] = None
    for f in list(femalePreference.keys()):
        studentAssignment[f] = None

    while len(unmatched) > 0:
        m = unmatched.popleft()
        f = malePreference[m][0]

        if studentAssignment[f] is None:
            advisorAssignment[m] = f
            studentAssignment[f] = m
        elif femalePreference[f].index(studentAssignment[f]) < femalePreference[f].index(m):
            unmatched.append(m)
        else:
            dump = studentAssignment[f]
            advisorAssignment[dump] = None
            unmatched.append(dump)
            advisorAssignment[m] = f
            studentAssignment[f] = m
        malePreference[m].pop(0)

    return [advisorAssignment, studentAssignment]


def residentMatching(residentPreference, hospitalPreference, quota):
    appAssignment = {} # applicant id to hospital
    hospAssignment = {} # hospital id to ***list*** of applicants

    '''IMPLEMENT METHOD HERE'''
    unmatched = deque()
    for r in list(residentPreference.keys()):
        unmatched.append(r)
        appAssignment[r] = None
    for h in list(hospitalPreference.keys()):
        hospAssignment[h] = []

    while len(unmatched) > 0:
        r = unmatched.popleft()
        if len(residentPreference[r]) == 0:
            continue
        h = residentPreference[r][0]
        if r not in hospitalPreference[h]:
            residentPreference[r].pop(0)
            unmatched.append(r)
            continue

        if len(hospAssignment[h]) < quota[h]:
            appAssignment[r] = h
            hospAssignment[h].append(r)
        else:
            dump = None
            for matched in hospAssignment[h]:
                if hospitalPreference[h].index(r) < hospitalPreference[h].index(matched):
                    if dump is None or hospitalPreference[h].index(dump) < hospitalPreference[h].index(matched):
                        dump = matched

            if dump is None:
                unmatched.append(r)
            else:
                appAssignment[dump] = None
                unmatched.append(dump)
                appAssignment[r] = h
                hospAssignment[h].append(r)
        residentPreference[r].pop(0)

    return [appAssignment, hospAssignment]


def ttc(patientPreference, kidneyPreference):
    patientMatch = {} # patient id to kidney

    '''IMPLEMENT METHOD HERE'''
    initial_match = {}
    for d, p in kidneyPreference.items():
        initial_match[p] = d

    unmatched_patients = list(patientPreference.keys())
    curr_pref = {}
    for p in unmatched_patients:
        curr_pref[p] = 0

    while len(unmatched_patients) > 0:
        prev_size = len(unmatched_patients)
        record = {}
        for p in unmatched_patients:
            top_donor = patientPreference[p][curr_pref[p]]
            top_patient = kidneyPreference[top_donor]

            while top_patient not in unmatched_patients:
                curr_pref[p] += 1
                top_donor = patientPreference[p][curr_pref[p]]
                top_patient = kidneyPreference[top_donor]
            record[p] = top_patient

        top_cycle = set()
        for p in unmatched_patients:
            cycle = set()
            cycle.add(p)
            target = record[p]

            while target not in cycle or target not in unmatched_patients:
                cycle.add(target)
                target = record[target]
            if target == p and len(cycle) > len(top_cycle):
                top_cycle = cycle

        for p in top_cycle:
            patientMatch[p] = initial_match[record[p]]
            unmatched_patients.remove(p)

        curr_size = len(unmatched_patients)
        if prev_size == curr_size:
            break

    if len(unmatched_patients) > 0:
        for p in unmatched_patients:
            patientMatch[p] = initial_match[p]

    return patientMatch


def saveFile(obj, algorithmName):
    with open(os.getcwd()+"/solution/"+algorithmName+".pickle", "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.algorithm == "stable_marriage":
        advisorPref = pickle.load(open(os.getcwd()+"/data/data_p1/advisor.pickle", 'rb'))
        studentPref = pickle.load(open(os.getcwd()+"/data/data_p1/student.pickle", 'rb'))
        solution = stableMarriage(advisorPref, studentPref)
    elif args.algorithm == "nrmp":
        residentPreference = pickle.load(open(os.getcwd()+"/data/data_p2/residents.pickle", 'rb'))
        hospitalPreference = pickle.load(open(os.getcwd()+"/data/data_p2/hospitals.pickle", 'rb'))
        quota = pickle.load(open(os.getcwd()+"/data/data_p2/quota.pickle", 'rb'))
        solution = residentMatching(residentPreference, hospitalPreference, quota)
    elif args.algorithm == "ttc":
        patientPreference = pickle.load(open(os.getcwd()+"/data/data_p3/patient.pickle", 'rb'))
        kidneyPreference = pickle.load(open(os.getcwd()+"/data/data_p3/kidney.pickle", 'rb'))
        solution = ttc(patientPreference, kidneyPreference)
    else:
        solution = None
    print(solution)
    if solution is not None:    
        saveFile(solution, args.algorithm)










