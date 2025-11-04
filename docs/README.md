# Documentation

This directory contains the project's documentation, planning materials, and Python package configurations. The purpose of this folder is to centralize all project-related documentation and planning resources to ensure smooth development and future updates.

- Add Markdown or PDF files as necessary.
- Structure subfolders for API docs or reports if needed.

## Folder Structure

- `planning/`: Planning and design documents for the project
- `reports/`: Cost Consideration report, containing analysis of financial and resource usage for cloud deployment
- `source/`:
- `python_package_planning`: Configuration file for the Python package setup
- `standards.md`: Defines project-wide naming conventions


## Files and Directories

### `planning/`

The `planning/` folder contains the planning and design documents for the project. It is intended for use by project stakeholders to track progress, plan future features, and align on technical decisions. This folder may include:

- **Project Roadmap**: High-level plans outlining the phases of the project.
- **Architecture Design**: Documents that describe the system architecture and any technical specifications.
- **Feature Planning**: Outlines for new features, user stories, and required tasks.
- **Sprint Evidencing Document**: A collection of notes from team meetings, including decisions made and follow-ups.

### `reports/`

The `reports/` directory holds cost-related documentation. It currently includes the Cost Consideration report, which outlines expected financial costs of running the cloud-based ML application using services like AzureML, DockerHub, MLflow, and Airflow. The report estimates weekly and monthly expenses and provides suggestions for managing costs effectively during development and deployment.

### `python_package_plan.ini`

This file contains the configuration and planning details for the Python package. It includes information about the package structure, dependencies, versioning, and metadata that is needed for the setup of the package. It is essential for setting up the environment and defining the structure of the project as a Python package.

## How to Use

- **Refer to `python_package_plan.ini`** to configure and set up the project package and dependencies.
- **Consult the documents in `planning/`** for insights into project planning, architecture decisions, and future work.
- Update planning documents regularly to track changes and milestones.

## Contributing to Documentation

If you wish to add new planning documents or make changes to the current ones, please follow these guidelines:

1. Create or modify documents inside the `planning/` folder.
2. For new configurations or major changes to the package setup, update `python_package_plan.ini`.
3. Ensure clarity and consistency in documentation to help team members easily understand the project's progress and next steps.
