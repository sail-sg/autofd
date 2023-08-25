SHELL          = /bin/bash
PROJECT_NAME   = autofd
PROJECT_FOLDER = autofd
PYTHON_FILES   = $(shell find . -type f -name "*.py")
CPP_FILES      = $(shell find . -type f -name "*.h" -o -name "*.cc")
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
PATH           := $(HOME)/go/bin:$(PATH)
ADDLICENSE_IGNORE =

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, yapf) && pip install 2to3 --upgrade

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format-11 || sudo apt-get install -y clang-format-11

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast)

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics

py-format: py-format-install
	yapf -r -d $(PYTHON_FILES)

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2023 $(ADDLICENSE_IGNORE) -check $(PROJECT_FOLDER)

lint: flake8 py-format

format: py-format-install addlicense-install
	yapf -ir $(PYTHON_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2023 $(ADDLICENSE_IGNORE) $(PROJECT_FOLDER)
