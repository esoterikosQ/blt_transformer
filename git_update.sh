#!/bin/bash
mv *.err *.out runtimelog
git add .
git commit -m "$1"
git push origin main 
