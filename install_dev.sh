#!/bin/bash

# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# GlaxoSmithKline LLC.  All rights reserved.  LLNL-CODE-784597
#
# OFFICIAL USE ONLY - EXPORT CONTROLLED INFORMATION
#
# PROTECTED CRADA INFORMATION - 7.31.19 - Authorized by: Jim Brase -
# CRADA TC02264
#
# This work was produced at the Lawrence Livermore National Laboratory (LLNL)
# under contract no. DE-AC52-07NA27344 (Contract 44) between the U.S. Department
# of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the
# operation of LLNL.  See license for disclaimers, notice of U.S. Government
# Rights and license terms and conditions.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

pip install -e . --user

