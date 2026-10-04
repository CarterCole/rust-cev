//! `#[derive(Choice)]` for fieldless enums.
//!
//! ```ignore
//! #[derive(cev::Choice, Debug, Clone, Copy, PartialEq)]
//! #[cev(instructions = "Which team should handle this ticket?", task = "ticket_router")]
//! enum Team {
//!     /// Payments, invoices and refunds
//!     Billing,
//!     /// Bugs, crashes and errors
//!     Tech,
//!     #[cev(rename = "sales_team", description = "New purchases")]
//!     Sales,
//! }
//! ```
//!
//! Variant names become option names (snake_case unless renamed), doc
//! comments become option descriptions, and variant order is the level order
//! when the enum is used as a score.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Lit, Meta, parse_macro_input};

#[proc_macro_derive(Choice, attributes(cev))]
pub fn derive_choice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(input) {
        Ok(t) => t.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[derive(Default)]
struct CevAttrs {
    rename: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    task: Option<String>,
}

fn cev_attrs(attrs: &[Attribute]) -> syn::Result<CevAttrs> {
    let mut out = CevAttrs::default();
    for a in attrs.iter().filter(|a| a.path().is_ident("cev")) {
        a.parse_nested_meta(|m| {
            let v: syn::LitStr = m.value()?.parse()?;
            let slot = if m.path.is_ident("rename") {
                &mut out.rename
            } else if m.path.is_ident("description") {
                &mut out.description
            } else if m.path.is_ident("instructions") {
                &mut out.instructions
            } else if m.path.is_ident("task") {
                &mut out.task
            } else {
                return Err(m.error("expected `rename`, `description`, `instructions` or `task`"));
            };
            *slot = Some(v.value());
            Ok(())
        })?;
    }
    Ok(out)
}

fn doc(attrs: &[Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) => Some(s.value().trim().to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let s = lines.join(" ").trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn snake(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let Data::Enum(e) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "Choice can only be derived for enums"));
    };
    if e.variants.len() < 2 {
        return Err(syn::Error::new_spanned(&input.ident, "a Choice enum needs at least two variants"));
    }
    let ty = &input.ident;
    let top = cev_attrs(&input.attrs)?;
    let mut idents = Vec::new();
    let mut names = Vec::new();
    let mut descs = Vec::new();
    for v in &e.variants {
        if !matches!(v.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(v, "Choice variants must not have fields"));
        }
        let a = cev_attrs(&v.attrs)?;
        idents.push(&v.ident);
        names.push(a.rename.unwrap_or_else(|| snake(&v.ident.to_string())));
        descs.push(match a.description.or_else(|| doc(&v.attrs)) {
            Some(d) => quote!(Some(#d)),
            None => quote!(None),
        });
    }
    let opt = |o: Option<String>| match o {
        Some(s) => quote!(Some(#s)),
        None => quote!(None),
    };
    let instructions = opt(top.instructions);
    let task = opt(top.task);
    let indices = 0..idents.len();
    let (ig, tg, wc) = input.generics.split_for_impl();
    Ok(quote! {
        impl #ig ::cev::Choice for #ty #tg #wc {
            const OPTIONS: &'static [::cev::OptionDef] = &[
                #( ::cev::OptionDef { name: #names, description: #descs } ),*
            ];
            const INSTRUCTIONS: Option<&'static str> = #instructions;
            const TASK: Option<&'static str> = #task;
            fn from_index(i: usize) -> Option<Self> {
                const ALL: &[fn() -> #ty] = &[ #( || #ty::#idents ),* ];
                ALL.get(i).map(|f| f())
            }
            fn index(&self) -> usize {
                match self { #( #ty::#idents => #indices ),* }
            }
        }
    })
}
