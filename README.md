
# Usage of this template

Please go through these steps before starting coding:

- [ ] **Create a new project**
    - on https://gitlab.cc-asp.fraunhofer.de/iis-scs-a
    - consider structure & naming guideline https://intern.iis.fhg.de/display/SCS/GitLab+structure
- [ ] **Copy current state of this template** to your new project
- [ ] **Activate GitLab runners**
    - Go to your project on GitLab web page -> Settings -> CI/CD -> Runners Expand -> check 'Enable instance runners for this project'
- [ ] **install unittest** in your python environment
- [ ] **install pylint and activate it** in your IDE
    - e.g. in Eclipse/PyDev it's: Window -> Settings -> PyDev -> Editor -> Code Analysis -> PyLint -> check "Use PyLint?"
	- **also set IDE to use given rcfile** (in Eclipse/PyDev in the same window as previous step insert "--rcfile=.pylintrc" in the arguments window)
- [ ] **install black and activate it on auto save** in your IDE
    - e.g. in Eclipse/PyDev it's: Window -> Settings -> PyDev -> Editor -> Code Style -> Code Formatter -> select "Black" as "Formatter style?"
    - AND also go to "Save Actions" and check "Auto-format editor contents before saving?" as well as "Sort imports on save?"
- [ ] **Rename src-subfolder** and **conda env name** in conda_env.yml to your projects name
- [ ] **Delete this template usage section** when you're done




# [projectname]

Code to [projectname]

Project deals with [...]

Project folder:
[...]

## Features
[...]

## installation

create conda env with all required packages
```bash
        conda env create -f conda_env.yml
```

## Usage

- put your import files into [...]
- run run_scenario.py
- get your results from [...]

## Authors and acknowledgment
Code and model by: [...]
Project knowledge: [...]



# License

## Used OSS libraries


| library         | min version    | checked version   | license |
----------------- | -------------- | ----------------- | ------------------- |
[...]             | [...]           | [...]              | [...] |



