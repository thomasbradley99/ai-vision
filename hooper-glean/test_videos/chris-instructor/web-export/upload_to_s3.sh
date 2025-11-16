#!/bin/bash
# Upload BJJ analysis to S3 for web viewing

BUCKET="end-nov-webapp-clann"
BASE_PATH="bjj-analysis/chris-instructor"

echo "📤 Uploading BJJ analysis to S3..."
echo "Bucket: $BUCKET"
echo "Path: $BASE_PATH"
echo ""

# Upload HTML viewer
echo "📄 Uploading viewer..."
aws s3 cp ../../../pipelines/tools/bjj_viewer.html s3://$BUCKET/$BASE_PATH/index.html

# Upload data files
echo "📊 Uploading data files..."
aws s3 cp ai_events_data.js s3://$BUCKET/$BASE_PATH/
aws s3 cp key_moments_data.js s3://$BUCKET/$BASE_PATH/
aws s3 cp observations_data.js s3://$BUCKET/$BASE_PATH/
aws s3 cp narrative_data.js s3://$BUCKET/$BASE_PATH/
aws s3 cp ground_truth_data.js s3://$BUCKET/$BASE_PATH/
aws s3 cp evaluation_data.js s3://$BUCKET/$BASE_PATH/

# Upload video (this takes a while)
echo "🎬 Uploading video..."
aws s3 cp ../../input/video.mov s3://$BUCKET/$BASE_PATH/video.mov

echo ""
echo "✅ Upload complete!"
echo ""
echo "🌐 View at:"
echo "https://$BUCKET.s3.eu-west-1.amazonaws.com/$BASE_PATH/index.html"
