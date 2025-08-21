# AWS Storage Integration Setup Guide

## Overview
This guide will help you integrate AWS S3 storage with your Smart Traffic Management System for cloud storage of videos, analytics data, and processed frames.

## Prerequisites
- AWS Account
- AWS IAM User with S3 permissions
- Python environment with boto3 installed

## Step 1: Install AWS Dependencies

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment (if using one)
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate

# Install AWS SDK
pip install boto3 botocore

# Or install all requirements
pip install -r requirements.txt
```

## Step 2: AWS Setup

### Create S3 Bucket (Option 1 - AWS Console)
1. Log in to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to S3 service
3. Click "Create bucket"
4. Enter bucket name: `smart-traffic-system-bucket` (or your preferred name)
5. Choose your preferred region
6. Keep default settings and create bucket

### Create IAM User (Option 2 - AWS Console)
1. Navigate to IAM service
2. Click "Users" → "Add users"
3. Username: `smart-traffic-user`
4. Access type: "Programmatic access"
5. Attach policy: "AmazonS3FullAccess" (or create custom policy below)
6. Save the Access Key ID and Secret Access Key

### Custom IAM Policy (Recommended for production)
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::smart-traffic-system-bucket",
                "arn:aws:s3:::smart-traffic-system-bucket/*"
            ]
        }
    ]
}
```

## Step 3: Configure Environment Variables

### Create `.env` file
```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your values
```

### Add AWS Configuration to `.env`
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=us-east-1
AWS_S3_BUCKET_NAME=smart-traffic-system-bucket
```

## Step 4: Test AWS Integration

### Start the backend server
```bash
python main.py
```

### Test AWS Status
```bash
# Check if AWS is working
curl http://localhost:5000/api/aws/status
```

Expected response:
```json
{
  "success": true,
  "aws_available": true,
  "stats": {
    "available": true,
    "bucket_name": "smart-traffic-system-bucket",
    "region": "us-east-1",
    "total_objects": 0,
    "total_size_mb": 0
  }
}
```

## Step 5: Features Available

### Automatic Backups
- **Processed Frames**: Every 10 frames during video processing
- **Analytics Data**: Every 50 frames during video processing
- **Database Backup**: Manual trigger via API or dashboard

### Manual Operations
- **Video Upload**: Upload videos via dashboard or API
- **File Management**: List, download, delete files
- **Presigned URLs**: Secure file access links

### API Endpoints
- `GET /api/aws/status` - Check AWS status and statistics
- `POST /api/aws/upload/video` - Upload video files
- `POST /api/aws/upload/analytics` - Upload analytics data
- `GET /api/aws/files` - List files in bucket
- `DELETE /api/aws/files/{key}` - Delete file
- `GET /api/aws/files/{key}/url` - Get presigned URL
- `POST /api/aws/backup/database` - Backup database

## Step 6: Frontend Usage

### Access AWS Storage Manager
1. Start the frontend: `npm run dev`
2. Log in to the dashboard
3. Click on "AWS Storage" tab
4. View status, upload files, manage storage

### Upload Videos
1. Go to AWS Storage → Upload tab
2. Select video file (.mp4, .avi, .mov, .mkv, .wmv)
3. Click "Upload Video"

### View Files
1. Go to AWS Storage → Files tab
2. Filter by type: All, Videos, Analytics, Frames
3. Download or delete files as needed

### Backup Database
1. Go to AWS Storage → Backup tab
2. Click "Backup Database"
3. Database file will be uploaded to S3

## Troubleshooting

### AWS Service Not Available
- Check `.env` file has correct AWS credentials
- Verify IAM user has S3 permissions
- Check S3 bucket exists and is accessible
- Verify network connectivity to AWS

### Upload Failures
- Check file size limits
- Verify file format is supported
- Check AWS credentials and permissions
- Monitor server logs for detailed errors

### Common Error Messages
- `AWS credentials not found` → Check `.env` file
- `Access Denied` → Check IAM permissions
- `Bucket not found` → Verify bucket name and region
- `Network timeout` → Check internet connection

## Cost Considerations

### S3 Pricing (approximate)
- **Storage**: $0.023 per GB/month
- **Requests**: $0.0004 per 1,000 PUT requests
- **Data Transfer**: First 1 GB free, then $0.09 per GB

### Estimated Costs for Traffic System
- **Small deployment**: ~$1-5/month
- **Medium deployment**: ~$10-25/month
- **Large deployment**: ~$50-100/month

### Cost Optimization Tips
- Use S3 Intelligent Tiering for automatic cost optimization
- Set lifecycle policies to move old data to cheaper storage classes
- Monitor usage with AWS Cost Explorer
- Consider using S3 Standard-IA for infrequently accessed data

## Security Best Practices

1. **Use IAM roles instead of access keys** (for EC2 deployment)
2. **Enable S3 bucket versioning** for data protection
3. **Set up S3 bucket logging** for audit trails
4. **Use VPC endpoints** for private access (enterprise)
5. **Rotate access keys regularly**
6. **Use presigned URLs** for temporary access
7. **Enable S3 encryption** at rest

## Next Steps

1. **Set up monitoring**: Use CloudWatch for S3 metrics
2. **Configure backups**: Set up automated S3 to Glacier archiving
3. **Add CDN**: Use CloudFront for faster file access
4. **Implement lifecycle policies**: Automatic data archiving
5. **Set up notifications**: SNS alerts for storage events

## Support

If you encounter issues:
1. Check the application logs
2. Verify AWS service status
3. Test AWS credentials using AWS CLI
4. Review IAM permissions
5. Check network connectivity

For AWS-specific issues, consult the [AWS Documentation](https://docs.aws.amazon.com/s3/).
