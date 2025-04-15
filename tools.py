from google.cloud import compute_v1
import os
class GCPVMTool:
    def init(self, project_id, zone):
        self.project_id = project_id
        self.zone = zone
        self.instance_client = compute_v1.InstancesClient()
    def create_vm(self, instance_name, machine_type, source_image, network="global/networks/default"):
        """
        Create a VM instance on GCP.
        Args:
            instance_name (str): Name of the VM instance.
            machine_type (str): Machine type (e.g., "e2-medium").
            source_image (str): Source image for the VM (e.g., "projects/debian-cloud/global/images/family/debian-11").
            network (str): Network to attach the VM to.
        Returns:
            str: Operation ID of the VM creation process.
        """
        instance = compute_v1.Instance()
        instance.name = instance_name
        instance.machine_type = f"zones/{self.zone}/machineTypes/{machine_type}"
        # Configure the disk
        disk = compute_v1.AttachedDisk()
        disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
            source_image=source_image
        )
        disk.auto_delete = True
        disk.boot = True
        instance.disks = [disk]
        # Configure the network interface
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = network
        instance.network_interfaces = [network_interface]
        # Create the VM
        operation = self.instance_client.insert(
            project=self.project_id,
            zone=self.zone,
            instance_resource=instance
        )
        return f"VM creation started. Operation ID: {operation.name}" 