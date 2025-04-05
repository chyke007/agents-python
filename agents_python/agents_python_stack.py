from typing_extensions import runtime
import os
from aws_cdk import (
    Duration,
    CustomResource,
    Stack,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
    aws_lambda as lambda_,
    aws_s3_notifications as s3n,
    aws_iam as iam,
    CfnOutput,
    aws_bedrock as bedrock,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_secretsmanager as secretsmanager
)
import json
from constructs import Construct
from lambdas.code import Lambdas


class AgentsPythonStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here
        account_id = os.environ["CDK_DEFAULT_ACCOUNT"]
        region = os.environ["CDK_DEFAULT_REGION"]

        # Load configuration
        with open('./agents_python/config.json', 'r') as config_file:
            config = json.load(config_file)

        # Define parameters
        agent_name = config['agentName']
        agent_alias_name = config['agentAliasName']
        knowledge_base_name = config['knowledgeBaseName']
        knowledge_base_description = config['knowledgeBaseDescription']
        s3_bucket_name = config['s3BucketName']+region+"-"+account_id
        agent_model_id = config['agentModelId']

        # Bedrock embedding model Amazon Titan Text v2
        embedding_model_id = config['embeddingModelId']
        embedding_model_arn = bedrock.FoundationModel.from_foundation_model_id(
            scope=self,
            _id='EmbeddingsModel',
            foundation_model_id=bedrock.FoundationModelIdentifier(embedding_model_id)).model_arn

        agent_description = config['agentDescription']
        agent_instruction = config['agentInstruction']
        agent_action_group_description = config['agentActionGroupDescription']
        agent_action_group_name = config['agentActionGroupName']
        table_name = config['dynamodbTableName']
        default_database_name = config['auroraDatabaseName']
        schema_table_name = config['auroraSchemaTableName']
        bedrock_user = config['bedrockUser']

        # Role that will be used by the KB
        kb_role = iam.Role(
            scope=self,
            id='AgentKBRole',
            role_name='AmazonBedrockExecutionRoleForKB',
            assumed_by=iam.ServicePrincipal('bedrock.amazonaws.com'))

        # Role that will be used by the Bedrock Agent
        agent_role = iam.Role(
            scope=self,
            id='AgentRole',
            role_name='AmazonBedrockExecutionRoleForAgents-HotelGenAI',
            assumed_by=iam.ServicePrincipal('bedrock.amazonaws.com'))

        base_lambda_policy = iam.ManagedPolicy.from_aws_managed_policy_name(
            managed_policy_name='service-role/AWSLambdaBasicExecutionRole')

        # Role that will be used by lambda function that Syncs Bedrock KB
        kb_lambda_role = iam.Role(
            scope=self,
            id='KBSyncLambdaRole',
            assumed_by=iam.ServicePrincipal(
                'lambda.amazonaws.com'),
            managed_policies=[base_lambda_policy])

        # Upload the dataset to Amazon S3
        s3Bucket = s3.Bucket(
            self, 'kbs3bucket',
            bucket_name=s3_bucket_name,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            event_bridge_enabled=True)
        bucket_deployment = s3_deploy.BucketDeployment(
            self, "Deploycontent",
            sources=[
                s3_deploy.Source.asset("./dataset")],
            destination_bucket=s3Bucket)

        # Grant permission to S3 to KB Role
        s3Bucket.grant_read(kb_role)

        # Allow KB permission to to invoke FM
        kb_role.add_to_policy(iam.PolicyStatement(
            sid='BedrockInvokeModelStatement',
            effect=iam.Effect.ALLOW,
            resources=[
                embedding_model_arn],
            actions=['bedrock:InvokeModel']))

        # Create a VPC for the Aurora cluster
        vpc = ec2.Vpc(self, "BedrockVPC", max_azs=2, nat_gateways=0)
        security_group = ec2.SecurityGroup(
            self, "SecurityGroup",
            vpc=vpc,
            allow_all_outbound=True
        )

        # Create a secret in Secrets Manager for the database credentials
        database_secret = secretsmanager.Secret(
            self,
            "AuroraDatabaseSecret",
            secret_name="AuroraDbCredentials",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=json.dumps({"username": bedrock_user}),
                generate_string_key="password",
                exclude_punctuation=True,
            ),
        )

        # Create an Aurora PostgreSQL V2 Serverless cluster
        aurora_cluster = rds.DatabaseCluster(
            self,
            "BedrockAuroraCluster",
            engine=rds.DatabaseClusterEngine.aurora_postgres(
                version=rds.AuroraPostgresEngineVersion.VER_14_10),
            vpc=vpc,
            default_database_name=default_database_name,
            credentials=rds.Credentials.from_secret(database_secret),
            serverless_v2_min_capacity=0.5,
            serverless_v2_max_capacity=1,
            writer=rds.ClusterInstance.serverless_v2(id="AuroraWriter",
                                                     instance_identifier="BedrockAuroraCluster-writer",
                                                     auto_minor_version_upgrade=False,
                                                     allow_major_version_upgrade=False,
                                                     publicly_accessible=False,
                                                     ),
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ),
            enable_data_api=True,
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Associate security to cluster
        aurora_cluster.connections.allow_default_port_from(security_group)

        # Grant necessary permissions to the Bedrock KB role
        aurora_cluster.grant_data_api_access(kb_role)

        Fn = Lambdas(self, "L")

        # Custom Lambda function to setup Postgres as a vector db: schema, table, etc
        pg_setup = CustomResource(self,
                                  "pg_setup",
                                  resource_type="Custom::PGSetup",
                                  service_token=Fn.table_creator.function_arn,
                                  properties=dict(
                                      cluster_arn=aurora_cluster.cluster_arn,
                                      secrets_arn=aurora_cluster.secret.secret_arn,
                                      table_name=schema_table_name,
                                      database_name=default_database_name,
                                      credentials_arn=database_secret.secret_arn,
                                  )
                                  )

        # Explicitly add dependency for the Settig up of Database
        # so pg_setup only gets run after the cluster and database secret setups are completed
        pg_setup.node.add_dependency(aurora_cluster)
        pg_setup.node.add_dependency(database_secret)

        Fn.table_creator.add_to_role_policy(iam.PolicyStatement(actions=["rds-data:ExecuteStatement"],
                                                                resources=[aurora_cluster.cluster_arn]))
        Fn.table_creator.add_to_role_policy(iam.PolicyStatement(actions=["secretsmanager:GetSecretValue"],
                                                                resources=[aurora_cluster.secret.secret_arn]))
        Fn.table_creator.add_to_role_policy(iam.PolicyStatement(actions=["secretsmanager:GetSecretValue"],
                                                                resources=[database_secret.secret_arn]))

        # Add Secrets Manager access to kb_role
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerAccess",
                effect=iam.Effect.ALLOW,
                actions=["secretsmanager:GetSecretValue"],
                resources=[database_secret.secret_arn,
                           database_secret.secret_full_arn],
            )
        )

        # Add RDS DescribeDBClusters permission to kb_role
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="RDSExecuteAccessStatement",
                effect=iam.Effect.ALLOW,
                actions=[
                    "rds:DescribeDBClusters"
                ],
                resources=[aurora_cluster.cluster_arn]
            )
        )

        # Create Knowledge Base
        knowledge_base = bedrock.CfnKnowledgeBase(
            scope=self,
            id='KBforAgent',
            name=knowledge_base_name,
            description=knowledge_base_description,
            role_arn=kb_role.role_arn,
            knowledge_base_configuration={'type': 'VECTOR',
                                          'vectorKnowledgeBaseConfiguration': {
                                              'embeddingModelArn': embedding_model_arn}
                                          },
            storage_configuration={
                'type': 'RDS',
                'rdsConfiguration': {
                    "credentialsSecretArn": database_secret.secret_full_arn,
                    'databaseName': default_database_name,
                    'resourceArn': aurora_cluster.cluster_arn,
                    'tableName': f'bedrock_integration.{schema_table_name}',
                    'fieldMapping': {
                        'metadataField': 'metadata',
                        'primaryKeyField': 'id',
                        'textField': 'chunks',
                        'vectorField': 'embedding'
                    }
                }
            }
        )

        # Explicitly add dependency for the creation of Knowledge Base
        # so it only gets created after the cluster and pg_setup setups are completed
        knowledge_base.node.add_dependency(aurora_cluster)
        knowledge_base.node.add_dependency(pg_setup)

        # Create Knowlegebase datasource for provisioned S3 bucket
        datasource = bedrock.CfnDataSource(
            scope=self,
            id=config['knowledgeBaseDataSourceId'],
            name=config['knowledgeBaseDataSourceName'],
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            data_source_configuration={'s3Configuration':
                                       {'bucketArn': s3Bucket.bucket_arn},
                                       'type': 'S3'},
            data_deletion_policy='RETAIN')

        # Create Lambda role to access KB
        kb_lambda_role.add_to_policy(iam.PolicyStatement(
            sid='SyncKBPolicy',
            effect=iam.Effect.ALLOW,
            resources=[
                knowledge_base.attr_knowledge_base_arn],
            actions=['bedrock:StartIngestionJob']))

        # Create Lambda function that sync Knowledge base and Datasource(S3)
        kb_sync_lambda = lambda_.Function(
            scope=self,
            id='SyncKB',
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset(
                'lambdas/kb_sync'),
            handler='lambda_function.handler',
            timeout=Duration.seconds(300),
            role=kb_lambda_role,
            environment={'KNOWLEDGE_BASE_ID': knowledge_base.attr_knowledge_base_id,
                         'DATA_SOURCE_ID': datasource.attr_data_source_id}
        )

        # Adds an event trigger to resync KB and Datasource when Datasource contents is updated
        s3Bucket.add_event_notification(s3.EventType.OBJECT_CREATED,
                                        s3n.LambdaDestination(kb_sync_lambda))
        s3Bucket.add_event_notification(s3.EventType.OBJECT_REMOVED,
                                        s3n.LambdaDestination(kb_sync_lambda))

        # Add an explicit dependency on the lambda, so that the bucket
        # deployment is started after the lambda is in place

        bucket_deployment.node.add_dependency(kb_sync_lambda)

        # Create the DynamoDB table
        dynamodbable = dynamodb.Table(self, config['dynamodbTableId'],
                                      partition_key=dynamodb.Attribute(
            name=config['dynamodbPartitionKeyId'],
            type=dynamodb.AttributeType.STRING
        ),
            table_name=table_name,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        agent_role.add_to_policy(iam.PolicyStatement(
            sid='RetrieveKBStatement',
            effect=iam.Effect.ALLOW,
            resources=[
                knowledge_base.attr_knowledge_base_arn],
            actions=['bedrock:Retrieve']))
        
        # Added support for cross-region inference
        agent_role.add_to_policy(
            iam.PolicyStatement(
                sid='InvokeInferenceProfile',
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel*", "bedrock:CreateInferenceProfile"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    "arn:aws:bedrock:*:*:inference-profile/*",
                    "arn:aws:bedrock:*:*:application-inference-profile/*",
                ],
            )
        )

        agent_role.add_to_policy(
            iam.PolicyStatement(
                sid='ListInferenceProfile',
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:GetInferenceProfile",
                    "bedrock:ListInferenceProfiles",
                    "bedrock:DeleteInferenceProfile",
                    "bedrock:TagResource",
                    "bedrock:UntagResource",
                    "bedrock:ListTagsForResource",
                ],
                resources=[
                    "arn:aws:bedrock:*:*:inference-profile/*",
                    "arn:aws:bedrock:*:*:application-inference-profile/*",
                ],
            )
        )

        action_group_function = lambda_.Function(
            self, "BedrockAgentActionGroupExecutor",
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset(
                'lambdas/actiongroup'),
            handler='lambda_function.lambda_handler',
            timeout=Duration.seconds(300)
        )
        action_group_function.add_to_role_policy(iam.PolicyStatement(
            sid="UpdateDynamoDB",
            effect=iam.Effect.ALLOW,
            resources=[
                dynamodbable.table_arn],
            actions=['dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:DeleteItem']))

        # Create the Agent
        cfn_agent = bedrock.CfnAgent(
            self, "CfnAgent",
            agent_name=agent_name,
            agent_resource_role_arn=agent_role.role_arn,
            auto_prepare=True,
            description=agent_description,
            foundation_model=agent_model_id,
            instruction=agent_instruction,
            idle_session_ttl_in_seconds=1800,
            knowledge_bases=[{'description': knowledge_base_description,
                              'knowledgeBaseId': knowledge_base.attr_knowledge_base_id}],

            action_groups=[bedrock.CfnAgent.AgentActionGroupProperty(
                action_group_name=agent_action_group_name,
                description=agent_action_group_description,

                # the properties below are optional
                action_group_executor=bedrock.CfnAgent.ActionGroupExecutorProperty(
                    lambda_=action_group_function.function_arn
                ),

                function_schema=bedrock.CfnAgent.FunctionSchemaProperty(
                    functions=[bedrock.CfnAgent.FunctionProperty(
                        name=config['func_getbooking_name'],
                        # the properties below are optional
                        description=config['func_getbooking_description'],
                        parameters={
                            config['func_getbooking_id']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The ID of the booking to retrieve",
                                required=True
                            )
                        }
                    ),
                        # create_booking
                        bedrock.CfnAgent.FunctionProperty(
                        name=config['func_createbooking_name'],
                        # the properties below are optional
                        description=config['func_createbooking_description'],
                        parameters={
                            config['func_createbooking_date']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The date of the booking",
                                required=True
                            ),
                            config['func_createbooking_person_name']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The name of the booking",
                                required=True
                            ),
                            config['func_createbooking_time']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The time of the booking",
                                required=True
                            ),
                            config['func_createbooking_num_guests']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="integer",

                                # the properties below are optional
                                description="The number of guests in the booking",
                                required=True
                            ),
                            config['func_createbooking_food']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The desired food",
                                required=False
                            )}
                    ),
                        # delete_booking
                        bedrock.CfnAgent.FunctionProperty(
                        name=config['func_deletebooking_name'],
                        # the properties below are optional
                        description=config['func_deletebooking_description'],
                        parameters={
                            config['func_deletebooking_id']: bedrock.CfnAgent.ParameterDetailProperty(
                                type="string",

                                # the properties below are optional
                                description="The ID of the booking to delete",
                                required=True
                            )
                        })
                    ]
                ),
            )])

        cfn_agent_alias = bedrock.CfnAgentAlias(
            self, "MyCfnAgentAlias",
            agent_alias_name=agent_alias_name,
            agent_id=cfn_agent.attr_agent_id)
    
        self.bedrock_agent_id = cfn_agent.attr_agent_id
        self.bedrock_agent_alias_id = cfn_agent_alias.attr_agent_alias_id

        lambda_.CfnPermission(
            self,
            "BedrockInvocationPermission",
            action="lambda:InvokeFunction",
            function_name=action_group_function.function_name,
            principal="bedrock.amazonaws.com",
            source_arn=cfn_agent.attr_agent_arn
        )

        # Agent is created with booking-agent-alias and prepared, so it shoudld be ready to test #

        # Declare the stack outputs
        CfnOutput(scope=self, id='S3_bucket', value=s3Bucket.bucket_name)
        CfnOutput(scope=self, id='Datasource_id',
                  value=datasource.attr_data_source_id)
        CfnOutput(scope=self, id='Knowedgebase_name',
                  value=knowledge_base.name)
        CfnOutput(scope=self, id='Knowedgebase_id',
                  value=knowledge_base.attr_knowledge_base_id)
        CfnOutput(scope=self, id='Agent_name', value=cfn_agent.agent_name)
        CfnOutput(scope=self, id='Agent_id', value=cfn_agent.attr_agent_id)
        CfnOutput(scope=self, id='Agent_alias_id', value=cfn_agent_alias.attr_agent_alias_id)

