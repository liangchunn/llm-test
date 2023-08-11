use std::{convert::Infallible, io::Write, path::PathBuf};

use clap::Parser;
use console::style;
use dialoguer::{theme::ColorfulTheme, Input};
use llm::{conversation_inference_callback, models::Llama, KnownModel};

const USER_NAME: &str = "USER:";
const CHARACTER_NAME: &str = "ASSISTANT:";

fn get_model(model_path: &PathBuf) -> Result<Llama, llm::LoadError> {
    let model_params = llm::ModelParameters {
        prefer_mmap: true,
        context_size: 2048,
        lora_adapters: None,
        use_gpu: true,
        gpu_layers: None,
        rope_overrides: None,
    };

    llm::load::<Llama>(
        model_path,
        llm::TokenizerSource::Embedded,
        model_params,
        |_| {},
    )
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut rng = rand::thread_rng();
    let model = get_model(&args.model_path)?;

    let model_path_str = args.model_path.display().to_string();
    let msg = format!("Model `{}` loaded!", model_path_str);
    println!("{}", style(msg).green().bright());

    let mut session = model.start_session(Default::default());

    loop {
        let prompt: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("")
            .interact_text()?;

        print!("👉");

        session
            .infer::<Infallible>(
                &model,
                &mut rng,
                &llm::InferenceRequest {
                    prompt: format!("{USER_NAME} {prompt}\n{CHARACTER_NAME}")
                        .as_str()
                        .into(),
                    parameters: &llm::InferenceParameters::default(),
                    play_back_previous_tokens: false,
                    maximum_token_count: None,
                },
                &mut Default::default(),
                conversation_inference_callback(&format!("{CHARACTER_NAME}"), |t| {
                    print!("{t}");
                    std::io::stdout().flush().unwrap()
                }),
            )
            .expect("error infering");

        println!();
        println!();
    }
}
