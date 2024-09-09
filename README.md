## Stack Diagram

![Stack Diagram](./diagrams/stack.png "A high-level overview of the stack")

## Module Overview

This table provides an overview of the purpose and responsibilities of each Python module in the repository.

| Module                   | Usage                                                                                         |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `train.py`               | Used to fine-tune a BERT model (`google/bert_uncased_L-2_H-128_A-2` from Hugging Face) for a downstream classification task. |
| `model.py`               | Configures the fine-tuning process, including defining which metrics to track and early stopping criteria. |
| `data.py`                | Fetches and prepares the required data for training and ensures it is in the correct format.  |
| `app.py`                 |  Sets up a FastAPI application to provide a web API for predictions using the `ColaONNXPredictor`. |
|                          | It includes endpoints for a home page and a prediction service that accepts text input and returns the prediction result. |
| `convert_model_to_onnx.py` | Converts the torch model into ONNX format. |
| `inference_onnx.py`     | Implements `ColaONNXPredictor`, which loads an ONNX model and |
|                         | provides text prediction, determining if the text is acceptable or unacceptable. |
| `utils.py`               | Collection of helper functions (e.g., `get_dvc_file_from_s3` which is used to get the tracked ONNX-formatted model from the s3 storage). |

## Directory Overview

This table provides an overview of the purpose of some directories.

| Module                   | Usage                                                                                         |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `.dvc`               | Created once you run dvc init. It contains the remote storage configurations. |
| `.github/workflows`  | Contains the GitHub workflows. |
| `configs`                | Contains configurations for preprocessing, training, and the transformer model to be used. |
| `k8s`                 |  Contains the manifest file for app deployment on Kubernetes, including endpoints for a home page and a prediction service that accepts text input to return prediction results. |
| `models` | Contains model.onnx.dvc, which is versioned here. |
| `monitoring`     | Refer to the `Monitoring` section below for details. |

## Continuous Integration and Deployment

This repository leverages GitHub Actions for automating the process of building and deploying Docker images to Docker Hub, followed by deploying the application to a Kubernetes cluster. The GitHub Actions workflow is defined within the `.github/workflows` directory.

### Workflow Trigger

- **Workflow Dispatch**: This workflow can be manually triggered via the GitHub UI, offering control over when the CI/CD process runs.

### Workflow Description

The workflow contains two jobs executed in sequential order:

#### publish_image
1. **Checkout Code**: The latest code is retrieved from the repository.
2. **Set up QEMU**: Prepares QEMU to emulate different architectures, enhancing compatibility.
3. **Set up Docker Buildx**: Configures Docker Buildx for more advanced build capabilities.
4. **Login to Docker Hub**: Authenticates to Docker Hub using credentials stored as GitHub secrets (`DOCKER_HUB_USERNAME`, `DOCKER_HUB_TOKEN`).
5. **Build and Push**: Builds the Docker image from the codebase and pushes it to Docker Hub. It utilizes AWS credentials (also stored as secrets) for actions requiring AWS resources.

#### deploy_to_k8s
1. **Dependency**: This job runs only after the successful completion of the `publish_image` job.
2. **Checkout Code**: Pulls the latest changes from the repository.
3. **Set up kubectl**: Prepares `kubectl` for interacting with Kubernetes, configured with credentials provided via GitHub secrets.
4. **Deploy**: Executes a deployment to Kubernetes using specified manifests. It accurately updates or modifies the Kubernetes configuration as defined.

### Usage

To trigger this workflow:
1. Navigate to the **Actions** tab in the GitHub repository.
2. Select the workflow you wish to run.
3. Click on **Run workflow** dropdown button and confirm your input to initiate the process manually.

The process ensures that the Docker images are built with the most recent changes and correctly deployed in a secure and consistent manner, maintaining the integrity of the production environment.


## Monitoring
To monitor the app deployed on Kubernetes, I use Filebeat, Elasticsearch, and Kibana. Filebeat helps track the logs in each app container, forwarding them to Elasticsearch. Kibana then connects to Elasticsearch to search and visualize these logs. To deploy these components on Kubernetes, I followed the steps outlined below:
* Create a directory for each component/service in `monitoring/`.
* Run the following command (assuming you have Helm installed) to add a new chart repository called `elastic`. A chart repository is a collection of Helm charts (packages of pre-configured Kubernetes resources):

```bash
helm repo add elastic https://helm.elastic.co
```
* Run the following commands to retrieve the default configuration values for each application from the Elastic Helm repository and save them to a file named `values.yml`:

```bash
helm show values elastic/logstash > monitoring/logstash/values.yml
helm show values elastic/elasticsearch > monitoring/elasticsearch/values.yml
helm show values elastic/filebeat > monitoring/filebeat/values.yml
``` 
Please note that I made some changes to the configurations in these `yml` files where necessary. You can use them as they are for your deployment. For example, in `monitoring/filebeat/values.yml`, I replaced `filebeatConfig` with the following:
```yaml
filebeatConfig:
    filebeat.yml: |
    filebeat.inputs:
    - type: container
        paths:
        - /var/log/containers/*.log
        processors:
        - add_kubernetes_metadata:
            host: ${NODE_NAME}
            matchers:
            - logs_path:
                logs_path: "/var/log/containers/"

    output.elasticsearch:
        host: '${NODE_NAME}'
        hosts: '["https://${ELASTICSEARCH_HOSTS:elasticsearch-master:9200}"]'
        username: '${ELASTICSEARCH_USERNAME}'
        password: '${ELASTICSEARCH_PASSWORD}'
        protocol: https
        ssl.certificate_authorities: ["/usr/share/filebeat/certs/ca.crt"]
``` 
* Install the applications on Kubernetes using the following commands:

```bash
helm install --kubeconfig /path/to/kubeconfig.yml elasticsearch elastic/elasticsearch -f monitoring/elasticsearch/values.yml
helm install --kubeconfig /path/to/kubeconfig.yml filebeat elastic/filebeat -f monitoring/filebeat/values.yml
helm install --kubeconfig /path/to/kubeconfig.yml kibana elastic/kibana -f monitoring/kibana/values.yml  
```

* To access the Kibana UI, use the following commands to acquire the username and password. Use `kubectl port-forward` first to make the UI reachable from your localhost.  

```bash
kubectl --kubeconfig /path/to/kubeconfig.yml port-forward svc/kibana-kibana 8090:5601
```

```bash
kubectl --kubeconfig /path/to/kubeconfig.yml get secret elasticsearch-master-credentials -o jsonpath="{.data.username}" | base64 --decode
kubectl --kubeconfig /path/to/kubeconfig.yml get secret elasticsearch-master-credentials -o jsonpath="{.data.password}" | base64 --decode
```

## Acknowledgments

This repository is based on the original work [MLOps-Basics](https://github.com/graviraja/MLOps-Basics) by graviraja. I extend my gratitude to the original author and contributors for their contributions to MLOps practices.

The work presented in this repository builds upon the existing framework with improvements to adapt to the latest developments in Python packages and address issues while enhancing the MLOps stack used. My goal is to modernize the implementation to better align with current industry standards and practices while maintaining the spirit of the original project.