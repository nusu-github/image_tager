use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, RecommendExample, RecommendPointsBuilder,
    ScoredPoint, SearchParamsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;

use crate::Config;

pub struct QdrantWrapper {
    client: Qdrant,
}

impl QdrantWrapper {
    pub fn new() -> Result<Self> {
        let app_config = Config::new()?;
        let client = Qdrant::from_url(&app_config.qdrant_url)
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        Ok(Self { client })
    }

    // Collection operations
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let response = self.client.list_collections().await?;
        Ok(response.collections.into_iter().map(|c| c.name).collect())
    }

    pub async fn create_collection(&self, name: &str, vector_size: u64) -> Result<()> {
        let config = VectorParamsBuilder::new(vector_size, Distance::Cosine);
        self.client
            .create_collection(CreateCollectionBuilder::new(name).vectors_config(config))
            .await?;
        Ok(())
    }

    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        self.client.delete_collection(name).await?;
        Ok(())
    }

    pub async fn get_collection_info(&self, name: &str) -> Result<String> {
        let info = self.client.collection_info(name).await?;
        Ok(format!("{info:#?}"))
    }

    // Point operations
    pub async fn add_points(&self, collection_name: &str, points: Vec<PointStruct>) -> Result<()> {
        self.client
            .upsert_points_chunked(
                UpsertPointsBuilder::new(collection_name, points).wait(true),
                32,
            )
            .await?;
        Ok(())
    }

    pub async fn search_points(
        &self,
        collection_name: &str,
        vector: Vec<Vec<f32>>,
        search_params: &SearchParams,
    ) -> Result<Vec<Payload>> {
        let builder = self.build_search_builder(collection_name, vector, search_params);
        let search_result = self.client.recommend(builder).await?;

        Ok(search_result
            .result
            .iter()
            .map(Self::convert_to_point_struct)
            .collect())
    }

    // Helper methods
    fn build_search_builder(
        &self,
        collection_name: &str,
        vector: Vec<Vec<f32>>,
        params: &SearchParams,
    ) -> RecommendPointsBuilder {
        vector
            .into_iter()
            .fold(
                RecommendPointsBuilder::new(collection_name, params.limit),
                |builder, vec| builder.add_positive(RecommendExample::from(vec)),
            )
            .with_payload(true)
            .params(
                SearchParamsBuilder::default()
                    .exact(params.exact)
                    .hnsw_ef(params.hnsw_ef),
            )
            .score_threshold(params.score_threshold)
    }

    fn convert_to_point_struct(scored_point: &ScoredPoint) -> Payload {
        let url = scored_point.payload.get("url").unwrap();
        let path = scored_point.payload.get("path").unwrap();
        let hash = scored_point.payload.get("hash").unwrap();
        Payload {
            path: path.to_string(),
            hash: hash.to_string(),
            url: url.to_string(),
        }
    }
}

pub struct SearchParams {
    pub score_threshold: f32,
    pub exact: bool,
    pub hnsw_ef: u64,
    pub limit: u64,
}

pub struct Payload {
    pub path: String,
    pub hash: String,
    pub url: String,
}
