terraform {
  required_providers {
    coder = {
      source = "coder/coder"
    }
    docker = {
      source = "kreuzwerker/docker"
    }
  }
}

# ---------------------------------------------------------------------------
# Interactive parameters — these render as the workspace creation form in
# the Coder dashboard.
# ---------------------------------------------------------------------------

data "coder_parameter" "gpu" {
  name         = "gpu"
  display_name = "Include GPU?"
  description  = "Enable a GPU-backed workspace for faster ECG autoencoder retraining. Not required just to run the demo — trained model artifacts are already committed."
  type         = "bool"
  default      = false
  mutable      = false
  order        = 1
}

data "coder_parameter" "component" {
  name         = "component"
  display_name = "Component to launch on start"
  description  = "Which part of CardioScope should boot automatically when the workspace starts."
  type         = "string"
  default      = "full-stack"
  mutable      = true
  order        = 2

  option {
    name  = "Full stack (backend + frontend)"
    value = "full-stack"
  }
  option {
    name  = "Backend only (FastAPI)"
    value = "backend-only"
  }
  option {
    name  = "Frontend only (React)"
    value = "frontend-only"
  }
  option {
    name  = "Notebooks only (JupyterLab)"
    value = "notebooks-only"
  }
}

data "coder_parameter" "dataset" {
  name         = "dataset"
  display_name = "Dataset subset to preload"
  description  = "Use a small bundled sample for a fast startup, or pull the full datasets if you plan to retrain models."
  type         = "string"
  default      = "sample"
  mutable      = false
  order        = 3

  option {
    name  = "Sample (fast startup)"
    value = "sample"
  }
  option {
    name  = "Full dataset (slower, requires data/ present)"
    value = "full"
  }
}

data "coder_parameter" "region" {
  name         = "region"
  display_name = "Deployment Region"
  description  = "Select the region to deploy your CardioScope workspace."
  type         = "string"
  default      = "us-east"
  mutable      = false
  order        = 4

  option {
    name  = "US East (N. Virginia)"
    value = "us-east"
  }
  option {
    name  = "EU West (London)"
    value = "eu-west"
  }
}

# ---------------------------------------------------------------------------
# Coder workspace / agent
# ---------------------------------------------------------------------------

data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

resource "coder_agent" "main" {
  os             = "linux"
  arch           = "amd64"
  startup_script = templatefile("${path.module}/startup.sh", {
    component = data.coder_parameter.component.value
    dataset   = data.coder_parameter.dataset.value
  })

  metadata {
    display_name = "CPU Usage"
    key          = "cpu"
    script       = "coder stat cpu"
    interval     = 10
    timeout      = 1
  }
  metadata {
    display_name = "Memory Usage"
    key          = "mem"
    script       = "coder stat mem"
    interval     = 10
    timeout      = 1
  }
}

# Expose the FastAPI backend and React dev server as workspace apps so they
# show up as clickable links in the Coder dashboard.
resource "coder_app" "backend" {
  agent_id     = coder_agent.main.id
  slug         = "backend"
  display_name = "CardioScope API"
  url          = "http://localhost:8000/docs"
  icon         = "/icon/database.svg"
  subdomain    = true
}

resource "coder_app" "frontend" {
  agent_id     = coder_agent.main.id
  slug         = "frontend"
  display_name = "CardioScope App"
  url          = "http://localhost:5173"
  icon         = "/icon/code.svg"
  subdomain    = true
}

resource "coder_app" "jupyter" {
  agent_id     = coder_agent.main.id
  slug         = "jupyter"
  display_name = "CardioScope Notebooks"
  url          = "http://localhost:8888"
  icon         = "/icon/jupyter.svg"
  subdomain    = true
}

# ---------------------------------------------------------------------------
# Underlying container — a plain Docker-based workspace running the agent.
# GPU flag (data.coder_parameter.gpu.value) can be wired to a `runtime =
# "nvidia"` block on this resource if your Coder deployment has GPU nodes
# available; left as a placeholder toggle here since GPU node pools are
# deployment-specific.
# ---------------------------------------------------------------------------

resource "docker_image" "cardioscope" {
  name = "python:3.11-slim"
}

resource "docker_container" "workspace" {
  count    = data.coder_workspace.me.start_count
  image    = docker_image.cardioscope.image_id
  name     = "coder-${data.coder_workspace_owner.me.name}-${data.coder_workspace.me.name}"
  hostname = data.coder_workspace.me.name
  command  = ["sh", "-c", coder_agent.main.init_script]
  env      = ["CODER_AGENT_TOKEN=${coder_agent.main.token}"]
}
