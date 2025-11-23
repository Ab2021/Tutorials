# Lab 07.9: Terraform Provisioners

## Objective
Use provisioners to execute scripts during resource creation.

## Learning Objectives
- Use local-exec provisioner
- Use remote-exec provisioner
- Understand provisioner limitations
- Implement proper error handling

---

## Local-Exec

```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  
  provisioner "local-exec" {
    command = "echo ${self.public_ip} >> instances.txt"
  }
  
  provisioner "local-exec" {
    when    = destroy
    command = "echo 'Instance ${self.id} destroyed' >> log.txt"
  }
}
```

## Remote-Exec

```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  key_name      = "my-key"
  
  connection {
    type        = "ssh"
    user        = "ec2-user"
    private_key = file("~/.ssh/id_rsa")
    host        = self.public_ip
  }
  
  provisioner "remote-exec" {
    inline = [
      "sudo yum update -y",
      "sudo yum install -y httpd",
      "sudo systemctl start httpd"
    ]
  }
}
```

## File Provisioner

```hcl
provisioner "file" {
  source      = "app.conf"
  destination = "/tmp/app.conf"
}
```

## Success Criteria
✅ Local-exec working  
✅ Remote-exec configuring instances  
✅ Files transferred  

**Time:** 40 min
